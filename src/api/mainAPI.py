# ⚠️ preocupaciones

'''
1. Falta de validacion fuerte y feedback estructurado
Todo depende de que el JSON este bien hecho, de que el modelo tenga la salida esperada, 
de que el adaptador coincida. Pero no hay una capa de validacion fuerte que diga “el modelo no devuelve 
lo que el unpacker espera” o “el JSON esta mal formado”.
    Solucion: Añadir validacion con pydantic mas profunda para los outputs y una validacion 
cruzada entre JSON ↔ codigo al cargar un modelo.
'''

'''
2. Documentacion y descubribilidad
Nadie puede entender el programa sin abrir el codigo. No hay descripciones ni ejemplo de payloads en los endpoints.
    Solucion : Utilizar FastAPI Docs (http://localhost:8000/docs) usando Body(...), Form(...) y UploadFile(...)
bien anotados con descripciones.
'''

'''
3. Poca trazabilidad / logging
Cuando algo falle en produccion, por ejemplo porque un modelo devuelve un shape inesperado, no sabria a donde fallo.
Si no hay logs claros, va a ser un infierno debuguear.
    Solucion: Usar logging de Python con niveles (info, warning, error) en puntos como: carga, inferencia, adaptacion, etc.
'''

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
import base64
from pathlib import Path
from api.func.model_controller import ModelController

# Rutas absolutas relativas a este archivo (src/api/mainAPI.py → ../../)
_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = _ROOT / "models"
CONFIGS_DIR = _ROOT / "configs"

MODEL_EXTENSIONS = {".onnx", ".tflite", ".h5", ".keras", ".pt", ".pth"}

app = FastAPI(
    title="Sistema de Vision por Computadora",
    description="API para carga de modelos y ejecucion de inferencias sobre imagenes.",
    version="1.0.0"
)

# Electron carga desde file:// — sin esto todos los fetch() y WS fallan por CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = ModelController()

class ModelPathRequest(BaseModel):
    model_path: str

class SelectModelRequest(BaseModel):
    model_name: str

class ConfidenceUpdateRequest(BaseModel):
    value: float


def _find_model_file(model_name: str) -> str:
    """Busca en models/ el archivo con el basename indicado."""
    matches = [p for p in MODELS_DIR.glob(f"{model_name}.*")
               if p.suffix.lower() in MODEL_EXTENSIONS]
    if not matches:
        raise FileNotFoundError(
            f"No se encontró archivo de modelo para '{model_name}' en {MODELS_DIR}/")
    return str(matches[0])


def _draw_detections(img_bgr: np.ndarray, detections) -> None:
    """Dibuja bounding boxes [x1,y1,x2,y2,conf,cls] sobre imagen BGR (in-place)."""
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (0, 191, 255)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{int(cls)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# ════════════════════════════════════════
# 1a Listar modelos disponibles
# ════════════════════════════════════════

@app.get("/get_models")
def get_models():
    try:
        models = sorted(p.stem for p in CONFIGS_DIR.glob("*.json"))
        return {"models": models}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


# ════════════════════════════════════════
# 1b Seleccionar modelo por nombre
# ════════════════════════════════════════

@app.post("/select_model")
def select_model(data: SelectModelRequest):
    try:
        model_path = _find_model_file(data.model_name)
        controller.load_model(model_path)
        return {"status": "ok", "message": f"Modelo cargado: {data.model_name}"}
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"status": "error", "detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


# ════════════════════════════════════════
# 1c Cargar modelo por path directo
# ════════════════════════════════════════

@app.post("/model/load")
def load_model(data: ModelPathRequest):
    try:
        controller.load_model(data.model_path)
        return {"status": "ok", "message": f"Modelo cargado: {data.model_path}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


# ════════════════════════════════════════
# 2 Actualizar umbral de confianza
# ════════════════════════════════════════

@app.post("/config/confidence")
def update_confidence(data: ConfidenceUpdateRequest):
    controller.update_confidence(data.value)
    return {"status": "ok", "new_confidence": data.value}


# ════════════════════════════════════════
# 3 Enviar imagen y obtener deteccion
# ════════════════════════════════════════

@app.post("/predict")
async def run_inference(file: UploadFile = File(...)):
    if controller.predict_fn is None:
        return JSONResponse(status_code=400, content={"status": "error", "detail": "No hay modelo cargado. Llamar a /model/load primero."})
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        image_np = np.array(image)

        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3]

        result = controller.inference(image_np)
        return {"status": "ok", "detections": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


# ════════════════════════════════════════
# 4 Descargar modelo
# ════════════════════════════════════════

@app.post("/model/unload")
def unload_model():
    controller.unload_model()
    return {"status": "ok", "message": "Modelo descargado."}


# ════════════════════════════════════════
# 5 WebSocket streaming con inferencia
# ════════════════════════════════════════

@app.websocket("/video_stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            # Limpiar encabezado data-URL si viene con prefijo
            if "," in data:
                data = data.split(",", 1)[1]

            img_bytes = base64.b64decode(data)
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if img_bgr is None:
                continue

            if controller.predict_fn is not None:
                try:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    detections = controller.inference(img_rgb)
                    if detections:
                        _draw_detections(img_bgr, detections)
                except Exception:
                    pass  # Enviar frame sin anotar si falla la inferencia

            # Las cámaras web envían píxeles sin espejo — se aplica aquí para
            # que el usuario vea el efecto espejo natural en el outputCanvas
            img_bgr = cv2.flip(img_bgr, 1)

            _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_frame = base64.b64encode(buf.tobytes()).decode("utf-8")
            await websocket.send_text(b64_frame)

    except WebSocketDisconnect:
        pass
