import os
from .model_loader import ModelLoader
from .general.json_reader import loadModelConfig
from .general.transformers import buildPreprocessor, buildPostprocessor
from .general.adapters import generate_input_adapter, generate_output_adapter
from .general.utils import ReactiveOutputConfig
from .general.unpackers import build_unpacker

'''   
    Tiene que comportarce como un controlador del backend dependiente de los eventos del cliente.
    Debe ser capaz de: 
        1_ Cargar modelos 
        2_ Configurar propiedades 
        3_ Ejecutar en CPU, o en otro procesador
        4_ Liberar recursos

    Patron aplicado: Strategy.
'''
class ModelController:

    def __init__(self):
        self.predict_fn = None
        self.input_adapter = None
        self.output_adapter = None
        self.unpack_fn = None
        self.preprocess_fn = None
        self.letter_transformers = None
        self.postprocess_fn = None
        self.model_format = None
        self.config = None

    def load_model(self, model_path: str):
        self.model_format = os.path.splitext(model_path)[1].lower()
        self.config = loadModelConfig(model_path)
        self.config.output = ReactiveOutputConfig(**self.config.output.dict())

        self.predict_fn = ModelLoader.load(model_path, self.model_format)
        self.preprocess_fn, self.letter_transformers = buildPreprocessor(self.config)
        self.input_adapter = generate_input_adapter(self.config.input)
        self.unpack_fn = build_unpacker(self.config.output.output_tensor.output_format)
        self.output_adapter = generate_output_adapter(self.config.output.tensor_structure)
        self.postprocess_fn = buildPostprocessor(self.config, self.letter_transformers)

    def inference(self, img):
        pre = self.preprocess_fn(img)
        adapted_input = self.input_adapter(pre)
        raw_output = self.predict_fn(adapted_input)
        unpacked = self.unpack_fn(raw_output)
        adapted_output = [self.output_adapter(row) for row in unpacked]
        return self.postprocess_fn(adapted_output)      
    
    def update_confidence(self, new_threshold: float):
        self.config.output.confidence_threshold = new_threshold

    def unload_model(self):
        self.predict_fn = None
        self.input_adapter = None
        self.output_adapter = None
        self.unpack_fn = None
        self.preprocess_fn = None
        self.postprocess_fn = None
        self.model_format = None
        self.config = None


"""
    Se le quitan las responsabilidades de adaptar las entradas y salidas de la IA en uso a 
"model_loader", por lo que dentro de proximos commits se tendra que cortar esa dependencia. 
    Pasa a depender directamente del script 'adapters.py', el cual esta en produccion. Todavia
se debe cambiar la forma en la que lee los JSONs el programa (leer las anotaciones del script
de los adaptadores para entender como), terminar el adaptador de outputs, hacer el adaptador 
de inputs y cambiar el orden de ejecucion del controlador a:

1_ Obtener el frame---------------------------------------(por hacer) <-- mainAPI.py
2_ Preprocesar frame--------------------------------------(hecho)     <-- transformers.py
3_ Adaptar el preproceso generico a la IA especifica------(por hacer) <-- adapters.py
4_ Generar la inferencia----------------------------------(hecho)     <-- model_loader.py
5_ Adaptar raw_output al formato generico del controlador-(por hacer) <-- adapters.py
6_ Postprocesar detections--------------------------------(hecho)     <-- transformers.py
7_ Devolver al cliente------------------------------------(por hacer) <-- mainAPI.py

"""
