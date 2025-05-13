from forms.keras_load import KerasInterpreter as Keras
from forms.tflite_load import TfliteInterpreter as Tflite
from forms.onnx_load import OnnxInterpreter as Onnx

class ModelLoader:

    @staticmethod
    def load(model_path, model_format):
        if model_format == ".h5":
            return Keras.loader(model_path), Keras.input_adapter(), Keras.output_adapter()
        elif model_format == ".tflite":
            return Tflite.loader(model_path), Tflite.input_adapter(), Tflite.output_adapter()
        elif model_format == ".onnx":
            return Onnx.loader(model_path), Onnx.input_adapter(), Onnx.output_adapter()
        else:
            raise ValueError(f"Formato no soportado: {model_format}")
        