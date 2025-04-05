import os
from .model_loader import ModelLoader
from .general.utils import findConfigModel
'''   
    Tiene que comportarce como un controlador del backend de dependiente de los eventos del cliente.
    Debe ser capaz de: 
        1_ Cargar procesos (model_loader.py)
        2_ Configurar propiedades (JSON)
        3_ Ejecutar en CPU, o en otro procesador, (do_inference)
        4_ Liberar recursos

    Patron aplicado: Strategy.
    '''
class ModelController:

    def __init__(self):
        self.predict_fn = None
        self.format_model = None
        self.config = None

    def load_model(self, model_path: str):
        self.model_format = os.path.splitext(model_path)[1].lower()
        self.config = findConfigModel(model_path)
        self.predict_fn = ModelLoader.load(model_path, self.model_format)

    def inference(self, img):
        if self.predict_fn is None:
            raise RuntimeError("Model no cargado.")
        return self.predict_fn(img)