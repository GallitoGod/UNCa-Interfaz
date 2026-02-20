from .config_schema import RuntimeConfig
from .forms.keras_load import kerasLoader as Keras
from .forms.tflite_load import tfliteLoader as Tflite
from .forms.onnx_load import onnxLoader as Onnx

class Model_loader:

    @staticmethod
    def load(model_path, runtime: RuntimeConfig, logger=None):
        if runtime.backend == "tensorflow":
            return Keras(model_path, runtime, logger)
        elif runtime.backend == "tflite":
            return Tflite(model_path, runtime, logger)
        elif runtime.backend == "onnxruntime":
            return Onnx(model_path, runtime, logger)
        else:
            raise ValueError(f"Fallo al cargar el modelo: {runtime.backend}, {model_path}")
        