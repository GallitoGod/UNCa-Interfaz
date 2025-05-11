import os
from .model_loader import ModelLoader
from .general.json_reader import loadModelConfig
from .general.transformers import buildPreprocessor, buildPostprocessor
'''   
    Tiene que comportarce como un controlador del backend de dependiente de los eventos del cliente.
    Debe ser capaz de: 
        1_ Cargar modelos (model_loader.py)
        2_ Configurar propiedades (JSON)
        3_ Ejecutar en CPU, o en otro procesador
        4_ Liberar recursos

    Patron aplicado: Strategy.
'''
class ModelController:

    def __init__(self):
        self.predict_fn = None
        self.preprocess_fn = None
        self.letter_transformers = None
        self.postprocess_fn = None
        self.model_format = None
        self.config = None

    def load_model(self, model_path: str):
        self.model_format = os.path.splitext(model_path)[1].lower()
        self.config = loadModelConfig(model_path)

        self.predict_fn = ModelLoader.load(model_path, self.model_format)
        self.preprocess_fn, self.letter_transformers = buildPreprocessor(self.config)
        self.postprocess_fn = buildPostprocessor(self.config, self.letter_transformers)

    def inference(self, img, confidence_override: float = None):
        preprocessed = self.predict_fn(img)
        raw_output = self.predict_fn(preprocessed)
        return self.postprocess_fn(raw_output)      # <--- TODAVIA HAY QUE ALTERAR LA CONFIANZA
    
    def unload_model(self):     # <--- SI TERMINA SIENDO INECESARIO, SE VA A BORRAR
        self.predict_fn = None
        self.preprocess_fn = None
        self.postprocess_fn = None
        self.model_format = None
        self.config = None