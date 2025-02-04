from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import cv2
import io
import os
from PIL import Image
import base64
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import onnxruntime as ort

app = FastAPI()

# Habilitar CORS para permitir que Electron acceda a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restringir a "http://localhost:3000" ??
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes: bytes, img_size: int = 224):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((img_size, img_size))
    image_array = np.array(image) / 255.0 
    return np.expand_dims(image_array, axis=0) 

@app.post("/predict")
async def predict(data: dict):

    try:
        image_base64 = data.get("image")
        print(f"Base64 recibido: {image_base64[:100]}...")
        model_type = data.get("model_type", "tf")

        image_data = base64.b64decode(image_base64.split(",")[1])
        processed_image = preprocess_image(image_data)

        if model_type == "tf":
            model = tf.keras.models.load_model("./models/tf_model.h5")
            prediction = model.predict(processed_image).tolist()
        elif model_type == "onnx":
            session = ort.InferenceSession("./models/onnx_model.onnx")
            input_name = session.get_inputs()[0].name
            prediction = session.run(None, {input_name: processed_image.astype(np.float32)})[0].tolist()
        else:
            raise HTTPException(status_code=400, detail="Modelo no soportado")

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(e)}")



@app.get("/models")
async def get_models():
    try:
        models = os.listdir("./models")
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar modelos: {str(e)}")

