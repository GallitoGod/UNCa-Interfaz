import os
import numpy as np
from .logger import setup_model_logger
from .reader_pipeline import Model_loader
from .reader_pipeline import load_model_config
from .reader_pipeline import Reactive_output_config
from .input_pipeline import build_preprocessor, generate_input_adapter
from .output_pipeline import unpack_out, buildPostprocessor, generate_output_adapter

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

    # TODO: el controlador debe terminar siendo un administrador de pipelines, nod debe conocer los detalles de cada pipeline
    # como ya lo hace. Debe entender el JSON, percatar si es calsificacion, deteccion, etc y en base a ello elegir las pipelines correctas.
    # Por lo que la siguiente actualizacion del controlador va a ser una reestructuracion completa para lograr cumplir ese objetivo.
    '''
        EL CONCEPTO:

        def inference(self, img):
            # despues de entender que clase de IA es:
            pre = InputPipeline.preprocess(img) #hace absolutamente todo el preprocesamiento
            infer = ModelLoader.predict(pre) #hace la inferencia
            return OutputPipeline.postprocess(infer) #hace el postprocesamiento y devuelve el resultado final
    '''
    def __init__(self):
        self.predict_fn = None
        self.input_adapter = None
        self.output_adapter = None
        self.unpack_fn = None
        self.preprocess_fn = None
        self.postprocess_fn = None
        self.model_format = None
        self.config = None
        self.logger = None

    def load_model(self, model_path: str):
        try:
            self.model_format = os.path.splitext(model_path)[1].lower()
            self.config = load_model_config(model_path)
            self.config.output = Reactive_output_config(**self.config.output.model_dump())

            self.predict_fn = Model_loader.load(model_path, self.model_format)
            self.preprocess_fn = build_preprocessor(self.config.input, self.config.runtime)
            self.input_adapter = generate_input_adapter(self.config.input)
            self.unpack_fn = unpack_out(self.config.output)
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
            unpacked = self.unpack_fn(raw_output, getattr(self.config, "runtime", None))
            if not isinstance(unpacked, (list, tuple)):
                unpacked = [unpacked]
            adapted_output = [self.output_adapter(list(r)) for r in unpacked]
            if not isinstance(adapted_output, (list, tuple)):
                adapted_output = [adapted_output]
            iw, ih = self.config.runtime.input_width,  self.config.runtime.input_height
            ow, oh = self.config.runtime.orig_width,   self.config.runtime.orig_height
            md = self.config.runtime.metadata_letter or {}
            self.logger.debug("[DBG] input/orig: input=%dx%d orig=%dx%d", iw, ih, ow, oh)
            self.logger.debug("[DBG] letter: %s", md)

            # ver 3 cajas ANTES del undo (espacio del tensor, 0..W_in/0..H_in)
            self.logger.debug(f"[DBG] tensor-space (pre-undo): {np.asarray(adapted_output[:3])}")
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
1_ Obtener el frame---------------------------------------(hecho) <-- mainAPI.py
2_ Preprocesar frame--------------------------------------(hecho) <-- transformers.py
3_ Adaptar el preproceso generico a la IA especifica------(hecho) <-- adapters.py
4_ Generar la inferencia----------------------------------(hecho) <-- model_loader.py
5_ Adaptar raw_output al formato generico del controlador-(hecho) <-- adapters.py
6_ Postprocesar detections--------------------------------(hecho) <-- transformers.py
7_ Devolver al cliente------------------------------------(hecho) <-- mainAPI.py
"""
