# 💡 Ideas:

# 💡 Falta logging del ciclo de vida de la IA activa
# Seria una buena idea agregar un sistema de log por modelo activo. Aunque sea en consola o un archivo .log,
# ayuda muchísimo en debugging y al implementar adaptadores.

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from PIL import Image
from func.model_controller import ModelController

app = FastAPI()
controller = ModelController()

# ════════════════════════════════════════
# 1 Cargar modelo
# ════════════════════════════════════════

@app.post("/load_model")
def load_model(model_path: str = Form(...)):
    try:
        controller.load_model(model_path)
        return {"status": "ok", "message": f"Modelo cargado: {model_path}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


# ════════════════════════════════════════
# 2 Actualizar umbral de confianza
# ════════════════════════════════════════

@app.post("/config/confidence")
def update_confidence(value: float = Form(...)):
    controller.update_confidence(value)
    return {"status": "ok", "new_confidence": value}


# ════════════════════════════════════════
# 3 Enviar imagen y obtener deteccion
# ════════════════════════════════════════

async def run_inference(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        image_np = np.array(image)

        # El controlador se encarga del color_order
        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3]

        result = controller.inference(image_np)
        return {"status": "ok", "detections": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


# ════════════════════════════════════════
# 4 Descargar modelo (de momento no va a andar)
# ════════════════════════════════════════

@app.post("/unload")
def unload_model():
    controller.unload_model()
    return {"status": "ok", "message": "Modelo descargado."}
