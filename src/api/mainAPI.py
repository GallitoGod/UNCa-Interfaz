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

from collections import deque
import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
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

# Config mutable de dibujo; se actualiza via POST /config/colors
_draw_config: dict = {
    "bbox_color":  (0, 191, 255),  # BGR equivalente de #00BFFF
    "label_color": (0,   0,   0),  # negro para texto sobre bbox
}

# Ultimos 50 errores de inferencia (in-memory)
_inference_errors: deque = deque(maxlen=50)


class ModelPathRequest(BaseModel):
    model_path: str

class SelectModelRequest(BaseModel):
    model_name: str

class ConfidenceUpdateRequest(BaseModel):
    value: float

class DrawConfigRequest(BaseModel):
    bbox_color:  str   # hex, ej: "#00BFFF"
    label_color: str   # hex, ej: "#FFFFFF"


def _hex_to_bgr(hex_color: str) -> tuple:
    """Convierte color hex (#RRGGBB) a tupla BGR para OpenCV."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _find_model_file(model_name: str) -> str:
    """Busca en models/ el archivo con el basename indicado."""
    matches = [p for p in MODELS_DIR.glob(f"{model_name}.*")
               if p.suffix.lower() in MODEL_EXTENSIONS]
    if not matches:
        raise FileNotFoundError(
            f"No se encontro archivo de modelo para '{model_name}' en {MODELS_DIR}/")
    return str(matches[0])


def _draw_detections(img_bgr: np.ndarray, detections) -> None:
    """Dibuja bounding boxes [x1,y1,x2,y2,conf,cls] sobre imagen BGR (in-place)."""
    bbox_color  = _draw_config["bbox_color"]
    label_color = _draw_config["label_color"]
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), bbox_color, 2)
        label = f"{int(cls)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 2, y1), bbox_color, -1)
        cv2.putText(img_bgr, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)

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
# 2b Actualizar colores de deteccion
# ════════════════════════════════════════

@app.post("/config/colors")
def update_colors(data: DrawConfigRequest):
    try:
        _draw_config["bbox_color"]  = _hex_to_bgr(data.bbox_color)
        _draw_config["label_color"] = _hex_to_bgr(data.label_color)
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "detail": str(e)})


# ════════════════════════════════════════
# 3 Descargar modelo
# ════════════════════════════════════════

@app.post("/model/unload")
def unload_model():
    controller.unload_model()
    return {"status": "ok", "message": "Modelo descargado."}


# ════════════════════════════════════════
# 4 WebSocket streaming con inferencia
# ════════════════════════════════════════

@app.websocket("/video_stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            if "," in data:
                data = data.split(",", 1)[1]

            img_bytes = base64.b64decode(data)
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if img_bgr is None:
                continue

            # Flipear antes de inferir y dibujar para que los labels queden legibles
            # Los labels muestran la clase en numeros, el sistema no sabe que son cada numero.
            # No se como solucionar eso.
            img_bgr = cv2.flip(img_bgr, 1)

            if controller.predict_fn is not None:
                try:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    detections = controller.inference(img_rgb)
                    if detections:
                        _draw_detections(img_bgr, detections)
                except Exception as e:
                    _inference_errors.append({
                        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                        "error": str(e),
                    })
                    h, w = img_bgr.shape[:2]
                    cv2.rectangle(img_bgr, (0, 0), (w, 44), (0, 0, 180), -1)
                    cv2.putText(img_bgr, "ERROR DE INFERENCIA — consultar /logs/inference",
                                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_frame = base64.b64encode(buf.tobytes()).decode("utf-8")
            await websocket.send_text(b64_frame)

    except WebSocketDisconnect:
        pass


# ════════════════════════════════════════
# 5 Obtener logs de inferencia
# ════════════════════════════════════════

@app.get("/logs/inference")
def get_inference_logs():
    return {"logs": list(_inference_errors)}


# ════════════════════════════════════════
# 6 Metricas de rendimiento
# ════════════════════════════════════════

@app.get("/metrics")
def get_metrics():
    stats = controller.perf.stats()
    if stats is None:
        return {"status": "no_data", "metrics": None}
    return {"status": "ok", "metrics": stats}
