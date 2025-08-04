import os
from .logger import setup_model_logger
from api.func.model_loader import ModelLoader
from api.func.general.json_reader import loadModelConfig
from api.func.general.transformers import buildPreprocessor, buildPostprocessor
from api.func.general.adapters import generate_input_adapter, generate_output_adapter
from api.func.general.thresholds_updater import ReactiveOutputConfig
from api.func.general.unpackers import build_unpacker

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

    # TODO: encapsular preprocessor + input_adapter en un InputPipeline
    # TODO: acoplamiento a detalles internos (ej. self.letter_transformers expuesto)
    # TODO: desacoplar unload_model para que cada componente tenga su propio release
    # TODO: escape de la validación de estado (None checks)
    def __init__(self):
        self.predict_fn = None
        self.input_adapter = None
        self.output_adapter = None
        self.unpack_fn = None
        self.preprocess_fn = None
        #self.letter_transformers = None
        self.postprocess_fn = None
        self.model_format = None
        self.config = None
        self.logger = None

    def load_model(self, model_path: str):
        try:
            self.model_format = os.path.splitext(model_path)[1].lower()
            self.config = loadModelConfig(model_path)
            self.config.output = ReactiveOutputConfig(**self.config.output.dict())

            self.predict_fn = ModelLoader.load(model_path, self.model_format)
            self.preprocess_fn = buildPreprocessor(self.config.input, self.config.runtime)
            self.input_adapter = generate_input_adapter(self.config.input)
            self.unpack_fn = build_unpacker(self.config.output.output_tensor.output_format)
            self.output_adapter = generate_output_adapter(self.config.output.tensor_structure)
            self.postprocess_fn = buildPostprocessor(self.config.output, self.config.runtime)
            
            self.logger = setup_model_logger(os.path.basename(model_path).split(".")[0])
            self.logger.info("Modelo cargado correctamente")
        except Exception as e:
            self.logger.exception(f"{e}")


    def inference(self, img):
        try:
            pre = self.preprocess_fn(img)
            adapted_input = self.input_adapter(pre)
            raw_output = self.predict_fn(adapted_input)
            unpacked = self.unpack_fn(raw_output)
            adapted_output = [self.output_adapter(row) for row in unpacked]
            result = self.postprocess_fn(adapted_output) 
            self.logger.info(f"Inferencia ejecutada: {len(result)} detecciones")
            return result    
        except ValueError as e:
            self.logger.exception(f"{e}") 
    
    def update_confidence(self, new_threshold: float):
        try:
            self.config.output.confidence_threshold = new_threshold
            self.logger.info(f"Confianza actualizada a {self.config.output.confidence_threshold}.")
        except Exception as e:
            self.logger.exception("No se pudo actualizar el umbral de confianza.")

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
1_ Obtener el frame---------------------------------------(casi hecho) <-- mainAPI.py
2_ Preprocesar frame--------------------------------------(hecho)     <-- transformers.py
3_ Adaptar el preproceso generico a la IA especifica------(hecho) <-- adapters.py
4_ Generar la inferencia----------------------------------(hecho)     <-- model_loader.py
5_ Adaptar raw_output al formato generico del controlador-(hecho) <-- adapters.py
6_ Postprocesar detections--------------------------------(hecho)     <-- transformers.py
7_ Devolver al cliente------------------------------------(casi hecho) <-- mainAPI.py
"""
