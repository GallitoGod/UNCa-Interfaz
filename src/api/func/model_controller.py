# model_controller.py
import os
import time
import numpy as np
from .logger import setup_model_logger, PerfMeter, run_warmup, make_dummy_input
from .reader_pipeline import Model_loader
from .reader_pipeline import load_model_config
from .reader_pipeline import Reactive_output_config
from .input_pipeline import build_preprocessor, generate_input_adapter
from .output_pipeline import buildPostprocessor, generate_output_adapter
from .output_pipeline.unpackers.registry import unpack_out

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

    # TODO: el controlador debe terminar siendo un administrador de pipelines, no debe conocer los detalles de cada pipeline
    # como ya lo hace. Debe entender el JSON, percatar si es clasificacion, deteccion, etc y en base a ello elegir las pipelines correctas.
    # Por lo que la siguiente actualizacion del controlador va a ser una reestructuracion completa para lograr cumplir ese objetivo.

    # TODO [INSTANCIA]: Segmentacion de instancias (ej: YOLOv8-seg, Mask R-CNN).
    #   A diferencia de la segmentacion semantica, combina deteccion de objetos + mascara por instancia.
    #   Requiere correr dos postprocesados en cadena:
    #     1_ Postprocesado de deteccion: obtener boxes + coeficientes de mascara por deteccion.
    #     2_ Postprocesado de mascara: combinar coeficientes con prototipos (YOLOv8-seg) o recortar mascaras
    #        directamente por ROI (Mask R-CNN). Binarizar con mask_threshold.
    #   Campos adicionales en el schema: mask_threshold (float), mask_channels (int, ej: 32 en YOLOv8-seg).
    #   El controlador deberia detectar model_type == "instance_segmentation" y despachar ambas pipelines.
    '''
        EL CONCEPTO:

        def inference(self, img):
            # despues de entender que clase de IA es:
            pre = InputPipeline.preprocess(img) #hace absolutamente todo el preprocesamiento
            infer = ModelLoader.predict(pre) #hace la inferencia
            return OutputPipeline.postprocess(infer) #hace el postprocesamiento y devuelve el resultado final
    '''

    '''
    "Implementacion de un modulo de instanciacion dinamica guiado por gramaticas libres de contexto. 
    El sistema utiliza esquemas formales para generar de forma determinista los contratos de pre y postprocesamiento
      de los tensores para cualquier Red Neuronal. Esto desacopla la logica de inferencia del framework subyacente, garantizando
      la validez semantica en tiempo de ejecucion y logrando una arquitectura de software totalmente agnostica."

    Abstraccion de Modelos: Desacoplamiento total entre el motor de UNCALens y las topologías de las RNC.

    Generacion Determinista: Uso de reglas de produccion formales para estructurar las configuraciones sin errores de sintaxis.

    Validacion Semantica: Prevencion de fallos de memoria o de dimensiones de tensores en GPU al asegurar la 
      integridad del archivo antes de la carga.

    Arquitectura Agnostica: Estandarizacion de la entrada/salida sin importar el origen del modelo (ONNX, TFLite, Keras, etc.).
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
        self.perf = PerfMeter(window=300)
        self._frame_idx = 0
        self._log_every = 60

    def load_model(self, model_path: str):
        try:
            self.perf.reset()
            self.logger = setup_model_logger(os.path.basename(model_path).split(".")[0])

            self.model_format = os.path.splitext(model_path)[1].lower()
            self.config = load_model_config(model_path)
            self.config.output = Reactive_output_config(**self.config.output.model_dump())

            self.predict_fn = Model_loader.load(model_path, self.config.runtime, self.logger)
            self.preprocess_fn = build_preprocessor(self.config.input, self.config.runtime)
            self.input_adapter = generate_input_adapter(self.config.input)
            w = self.config.runtime.warmup
            if w.enabled and w.runs > 0:
                dummy_input = make_dummy_input(self.preprocess_fn, self.input_adapter, self.config)
                run_warmup(self.predict_fn, dummy_input, runs=w.runs, logger=self.logger)

            self.unpack_fn = unpack_out(self.config.output)
            self.output_adapter = generate_output_adapter(self.config.output.tensor_structure)
            self.postprocess_fn = buildPostprocessor(self.config.output, self.config.runtime)
            
            self.logger.info("Modelo cargado correctamente.")
            self.logger.debug(
                f"""Caracteristicas:
                Input:
                    Layout: {self.config.input.input_str.layout}
                    dtype:  {self.config.input.input_str.dtype}
                Output:
                    box format: {self.config.output.tensor_structure.box_format}
                    unpacker: {self.config.output.pack_format}
                """)
        except Exception as e:
            self.logger.exception(e)



    def inference(self, img):
        t0 = time.perf_counter()

        t_pre0 = time.perf_counter()
        pre = self.preprocess_fn(img)
        adapted_input = self.input_adapter(pre)
        t_pre1 = time.perf_counter()

        t_inf0 = time.perf_counter()
        raw_output = self.predict_fn(adapted_input)
        t_inf1 = time.perf_counter()

        t_post0 = time.perf_counter()

        unpacked = self.unpack_fn(raw_output, getattr(self.config, "runtime", None))

        if isinstance(unpacked, (list, tuple)):
            if len(unpacked) == 0:
                unpacked = np.empty((0, 6), dtype=np.float32)
            elif len(unpacked) == 1 and hasattr(unpacked[0], "ndim"):
                unpacked = unpacked[0]
            else:
                raise ValueError(f"unpack_fn devolvio {len(unpacked)} outputs; normalizacion ambigua.")

        unpacked = np.asarray(unpacked)

        if unpacked.ndim == 3 and unpacked.shape[0] == 1:
            unpacked = unpacked[0]

        if unpacked.ndim == 1:
            unpacked = unpacked[None, :]
        # -----------------------------------------------

        # boxes_scores ya entrega [x1,y1,x2,y2,conf,cls] en formato estandar del sistema.
        # Aplicar el adapter encima reordena mal las coords (swapea x/y de vuelta a yxyx).
        # raw y yolo_flat si necesitan el adapter porque salen en el espacio del tensor sin reordenar.
        _NEEDS_ADAPTER = {"raw", "yolo_flat", "tflite_detpost", "anchor_deltas"}
        pack_fmt = (getattr(self.config.output, "pack_format", "raw") or "raw").lower()
        if pack_fmt in _NEEDS_ADAPTER:
            adapted_output = [self.output_adapter(r) for r in unpacked]
        else:
            adapted_output = unpacked  # ya en [x1,y1,x2,y2,conf,cls]

        iw, ih = self.config.runtime.runtimeShapes.input_width,  self.config.runtime.runtimeShapes.input_height
        ow, oh = self.config.runtime.runtimeShapes.orig_width,   self.config.runtime.runtimeShapes.orig_height
        md = self.config.runtime.runtimeShapes.metadata_letter or {}
        self.logger.debug("[DBG] input/orig: input=%dx%d orig=%dx%d", iw, ih, ow, oh)
        self.logger.debug("[DBG] letter: %s", md)
        self.logger.debug(f"[DBG] tensor-space (pre-undo): {np.asarray(adapted_output[:3])}")

        result = self.postprocess_fn(adapted_output)
        self.logger.debug(f"{result[:5]}")
        self.logger.info(f"Inferencia ejecutada: {len(result)} detecciones")
        t_post1 = time.perf_counter()

        t1 = time.perf_counter()

        pre_ms   = (t_pre1 - t_pre0) * 1000
        inf_ms   = (t_inf1 - t_inf0) * 1000
        post_ms  = (t_post1 - t_post0) * 1000
        total_ms = (t1 - t0) * 1000

        self.perf.push(pre_ms, inf_ms, post_ms, total_ms)

        self._frame_idx += 1
        if self._frame_idx % self._log_every == 0:
            s = self.perf.stats()
            if s:
                self.logger.debug(
                    "PERF n=%d avg=%.2fms p95=%.2fms fps=%.2f | pre=%.2f inf=%.2f post=%.2f",
                    s["n"], s["avg_ms"], s["p95_ms"], s["fps_avg"],
                    s["pre_avg_ms"], s["inf_avg_ms"], s["post_avg_ms"],
                )
        return result



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
