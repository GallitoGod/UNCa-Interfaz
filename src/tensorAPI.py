from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
from typing import Union
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import cv2
import os

app = FastAPI()

# Habilitar CORS para permitir que Electron acceda a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restringir a "http://localhost:3000" ??
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loaded_model = {"name": None, "model": None}

def load_model(model_name):
    model_path = f"./models/{model_name}"

    if loaded_model["name"] == model_name:
        print(f"Modelo '{model_name}' ya está en uso.")
        return loaded_model["model"]

    if loaded_model["model"] is not None:
        print(f"Liberando modelo anterior: {loaded_model['name']}")
        loaded_model["model"] = None

    if model_name.endswith(".h5"):
        print(f"Cargando modelo TensorFlow: {model_name}")
        loaded_model["model"] = tf.keras.models.load_model(model_path)
    elif model_name.endswith(".onnx"):
        print(f"Cargando modelo ONNX: {model_name}")
        loaded_model["model"] = ort.InferenceSession(model_path)
    else:
        raise HTTPException(status_code=400, detail="Formato de modelo no soportado")

    loaded_model["name"] = model_name
    return loaded_model["model"]

def preprocess_image(image_bytes: bytes, img_size: int = 224):
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

@app.post("/predict")
async def predict(image_base64: str):

    try:
        if loaded_model["name"] is None:
            raise HTTPException(status_code=400, detail="No hay modelo cargado")
        image_data = base64.b64decode(image_base64.split(",")[1])
        processed_image = preprocess_image(image_data)

        if loaded_model["name"].endswith(".h5"):
            prediction = loaded_model["model"].predict(processed_image).tolist()
        elif loaded_model["name"].endswith(".onnx"):
            input_name = loaded_model["model"].get_inputs()[0].name
            prediction = loaded_model["model"].run(None, {input_name: processed_image.astype(np.float32)})[0].tolist()
        else:
            raise HTTPException(status_code=400, detail="Error con el modelo cargado")

        return {"prediction": prediction}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(err)}")



@app.get("/get_models")
async def get_models():
    try:
        data = os.listdir("./models")
        models = []
        for model in data:
            if model.endswith((".onnx", ".h5")):
                models.append(model)
        if models  == []:
            raise HTTPException(status_code=400, detail="No hay modelos disponibles")
        return {"models": models}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error al listar modelos: {str(err)}")


@app.post("/select_model")
async def select_model(model_name: str):
    try:
        model_path = f"./models/{model_name}"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {model_name}")
        load_model(model_name)
        return {"message": f"Modelo cambiado a {model_name}"}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error al cambiar modelo: {str(err)}")