from forms.keras_load import kerasLoader
from forms.tflite_load import tfliteLoader
from forms.onnx_load import onnxLoader
from .general.config_schema import InputConfig
from typing import Callable
import numpy as np
import cv2

class ModelLoader:

    @staticmethod
    def load(model_path, model_format):
        if model_format == ".h5":
            return kerasLoader(model_path)
        elif model_format == ".tflite":
            return tfliteLoader(model_path)
        elif model_format == ".onnx":
            return onnxLoader(model_path)
        else:
            raise ValueError(f"Formato no soportado: {model_format}")
        
    
def buildPreprocessor(config: InputConfig) -> Callable[[np.ndarray], np.ndarray]:
    steps = []
    if config.letterbox:
        pass
        #   steps.append(lambda img: letterboxResize(img, (config.width, config.height)))   <--- ESTA ES LA IDEA CUANDO HAGA LA FUNCION 
    else:
        steps.append(lambda img: cv2.resize(img, (config.width, config.height)))

    if config.scale:
        steps.append(lambda img: img.astype(np.float32) / 255.0)

    if config.normalize:
        mean = np.array(config.mean).reshape(1, 1, -1)
        std = np.array(config.std).reshape(1, 1, -1)
        steps.append(lambda img: (img - mean) / std)

    if config.channels == 1:
        steps.append(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None])

    def preprocess(img):
        for step in steps:
            img = step(img)
        return img

    return preprocess

