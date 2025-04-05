from forms.keras_load import kerasLoader
from forms.tflite_load import tfliteLoader
from forms.onnx_load import onnxLoader

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