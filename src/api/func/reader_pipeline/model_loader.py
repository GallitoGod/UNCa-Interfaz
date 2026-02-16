from config_schema import RuntimeConfig
from api.forms.keras_load import kerasLoader as Keras
from api.forms.tflite_load import tfliteLoader as Tflite
from api.forms.onnx_load import onnxLoader as Onnx

class Model_loader:

    @staticmethod
    def load(model_path, runtime: RuntimeConfig):
        if runtime.backend == "tensorflow":
            return Keras(model_path)
        elif runtime.backend == "tflite":
            return Tflite(model_path)
        elif runtime.backend == "onnxruntime":
            return Onnx(model_path)
        else:
            raise ValueError(f"Fallo al cargar el modelo: {runtime.backend}, {model_path}")
        