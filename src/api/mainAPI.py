
"""
   1_ SOLUCIONAR EL CAMBIO DE CONFIANZA (ver script utils.py para entender la idea).
   2_ CREAR PARSERS PARA CAMBIAR LAS ENTRADAS DISTINTAS A LAS SOPORTADAS POR EL PROGRAMA.
   3_ CAMBIAR TODAS LAS APIs PARA QUE PUEDAN FUNCIONAR CON EL CONTROLADOR.
   4_ HACER TESTS PARA TODA LA APLICACION Y PARA DISTINTAS ENTRADAS Y SALIDAS DE MODELOS.
"""






#EL CODIGO DE TODO ESTE SCRIPT ESTA DESACTUALIZADO


from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import tensorflow as tf
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

loaded_model = {"name": None, "model": None, "predict_fn": None}

def preprocess_image(image_bytes: bytes, img_size: int = 224):
    # Funcion para procesar la imagen
    ''' IMPORTANTE:
        
        Posiblemente esta funcion necesite de muchos cambios por el hecho de que cada ia necesita 
    cosumir imagenes en distintos tamaños, normalizaciones, canales de color y hasta formatos de entrada.
    Por lo que esta funcion va a tener que ser flexible en la forma de procesar las imagenes. Una idea para hacerlo
    es utilizar un objeto, siendo la ia seleccionada en el cliente, el cual tenga todas las especificaciones necesarias
    para procesar bien las imagenes desde esta funcion (la cual incluso podria ser un metodo de la clase).
    '''
    image = np.frombuffer(image_bytes, dtype=np.uint8) # Convierte la imagen en un array de valores enteros
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) # La imagen se procesa en 3 canales (RGB/BGR)
    image = cv2.resize(image, (img_size, img_size)) # Redimensiona la imagen a 224x224
    image = image.astype(np.float32) / 255.0 # Convierte los valores de 0-255 a 0-1 (float32)
    return np.expand_dims(image, axis=0) # agrega una dimension al array para un batch valido



#   De esta forma no necesito la api predict y puedo manejar websocket
@app.websocket("/video_stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()  
            image_data = base64.b64decode(data.split(",")[1]) 
            processed_image = preprocess_image(image_data)  
            prediction = loaded_model["predict_fn"](processed_image)  

            await websocket.send_json({"prediction": prediction})  
    except Exception as e:
        print(f"Error en WebSocket: {e}")
        await websocket.close()





@app.get("/get_models")
async def get_models():
    '''
        El objetivo de la api "get_models" es dar todos los modelos disponibles en la carpeta "models"
    al cliente de la aplicacion.
    '''
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
    ''' 
        El objetivo de la api "select_model" es cargar un modelo especifico abstrayendo toda la logica
    dependiente de cada formato en una unico diccionario para su uso en la api "predict".
    '''
    try:
        model_path = f"./models/{model_name}"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {model_name}")
        
        ext = os.path.splitext(model_path)[1]
        loaded_model["name"] = model_name
        loaded_model["model"] = ext

        if ext.endswith(".h5"):
            try:
                model = tf.keras.models.load_model(model_path)
                predict_fn = lambda img: model.predict(img).tolist()
            except Exception as err:
                raise HTTPException(status_code=404, detail=f"Error al cargar el modelo: {err}")

        elif ext.endswith(".tflite"):
            try:    
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                def tflite_predict(img):
                    interpreter.set_tensor(input_details[0]['index'], img)
                    interpreter.invoke()
                    return interpreter.get_tensor(output_details[0]['index']).tolist()
                predict_fn = tflite_predict
            except Exception as err:
                raise HTTPException(status_code=404, detail=f"Error al cargar el modelo: {err}")
            
        elif ext.endswith(".onnx"):
            try:    
                session = ort.InferenceSession(model_path)
                input_name = session.get_inputs()[0].name
                predict_fn = lambda img: session.run(None, {input_name: img})[0].tolist()
            except Exception as err:
                raise HTTPException(status_code=404, detail=f"Error al cargar el modelo: {err}")

        elif ext.endswith(".pth") or ext.endswith(".pt"):
            try:
                import torch
                model = torch.load(model_path) if ext.endswith("pth") else torch.jit.load(model_path)
                model.eval()

                def torch_predict(img):
                    with torch.no_grad():
                        return model(torch.tensor(img)).numpy().tolist()
                predict_fn = torch_predict
            except Exception as err:
                raise HTTPException(status_code=404, detail=f"Error al cargar el modelo: {err}")

        else:
            raise HTTPException(status_code=400, detail="Formato de modelo no soportado")
        
        loaded_model["predict_fn"] = predict_fn
        return {"message": f"Modelo cambiado a {model_name}"}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error al cambiar modelo: {str(err)}")
    

@app.post("/predict")
async def predict(image_base64: str):
    '''
        El objetivo de la api "predict" es recibir una imagen en base64 y realizar una prediccion
    con la abstraccion hecha en la api "select_model".
    '''
    try:
        if loaded_model["name"] is None:
            raise HTTPException(status_code=400, detail="No hay modelo cargado")
        image_data = base64.b64decode(image_base64.split(",")[1]) # Decodifica imagen base64
        processed_image = preprocess_image(image_data) # Prepara la imagen para el consumo
        prediction = loaded_model["predict_fn"](processed_image) # predice el resultado de la imagen

        return {"prediction": prediction}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(err)}")
    
    '''
        Esta api debe ser modificada para enviar los resultados dibujados directamente al cliente, 
    OpneCV es una libreria que optimiza la generacion de imagenes utilizando arrays de pixeles directamente en C,
    por lo que es mas eficiente hacerlo en la api que en el cliente.
        Aparte se tiene que dar la capacidad de dar parametros de personalizacion de colores, grosores y hasta incluso fuentes
    si se da el caso.
        Por ultimo, codificar la imagen procesada en base64 y devolverla al cliente.
    
        IDEAS:
            - Usar argumentos opcionales en FastAPI para cambiar dinamicamente los colores y estilos.
            - Usar cv2.putText() para agregar texto a la imagen.
            - Conviertir el color de "255,0,0" a una tupla (255,0,0) usando .split(",) y map(int, x.split(","))
    '''