from forms.keras_load import kerasLoader as Keras
from forms.tflite_load import tfliteLoader as Tflite
from forms.onnx_load import onnxLoader as Onnx

class ModelLoader:

    @staticmethod
    def load(model_path, model_format):
        if model_format == ".h5":
            return Keras.loader(model_path)
        elif model_format == ".tflite":
            return Tflite.loader(model_path)
        elif model_format == ".onnx":
            return Onnx.loader(model_path)
        else:
            raise ValueError(f"Formato no soportado: {model_format}")
        