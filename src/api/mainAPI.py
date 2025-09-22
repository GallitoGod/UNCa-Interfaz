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

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from PIL import Image
from api.func.model_controller import ModelController

app = FastAPI(
    title="Sistema de Vision por Computadora",
    description="API para carga de modelos y ejecucion de inferencias sobre imagenes.",
    version="1.0.0"
)
controller = ModelController()

class ModelPathRequest(BaseModel):
    model_path: str

class ConfidenceUpdateRequest(BaseModel):
    value: float

# ════════════════════════════════════════
# 1 Cargar modelo
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
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        image_np = np.array(image)

        # El controlador se encarga del color_order
        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3]
        

        result = controller.inference(image_np)
        result = [det.tolist() for det in result]
        return {"status": "ok", "detections": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


# ════════════════════════════════════════
# 4 Descargar modelo (de momento no va a andar).
# ════════════════════════════════════════

@app.post("/model/unload")
def unload_model():
    controller.unload_model()
    return {"status": "ok", "message": "Modelo descargado."}
