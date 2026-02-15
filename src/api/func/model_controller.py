# model_controller.py
import os
import time
import numpy as np
from .logger import setup_model_logger, PerfMeter
from .reader_pipeline import Model_loader
from .reader_pipeline import load_model_config
from .reader_pipeline import Reactive_output_config
from .input_pipeline import build_preprocessor, generate_input_adapter
from .output_pipeline import buildPostprocessor, generate_output_adapter
from .output_pipeline.unpackers import unpack_out

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

    '''
    0) Instrumentacion minima (una sola vez por carga de modelo)    <--- COMPLETADO
        Loggear model_id, format (onnx/tflite/tf), input_shape, dtype, preprocess usado.
        Loggear tiempos por etapa: t_pre, t_inf, t_post, t_draw, t_total.
        Loggear fps_avg y p95 (o al menos promedio + peor caso).
    '''
    '''
    1) Config JSON: runtime controlado por modelo (NO hardcode)
        Agregar en schema/config algo asi (idea, no literal):
        runtime.device: "cpu" | "gpu"
        runtime.backend: "onnxruntime" | "tflite" | "tensorflow" (si aplica)
        runtime.threads: intra_op, inter_op, num_threads
        runtime.onnx.providers: lista ordenada (ej ["CUDAExecutionProvider","CPUExecutionProvider"])
        runtime.tflite.delegates: ["gpu"] o vacio
        runtime.warmup_runs, runtime.benchmark_runs
    '''
    '''
    2) ONNX Runtime: GPU habilitable y fallback
        En onnx_load.py (donde se crea InferenceSession):
        Detectar si el paquete es onnxruntime-gpu y si CUDAExecutionProvider esta disponible.
        Crear sesion con providers desde config:
            si device=="gpu": ["CUDAExecutionProvider","CPUExecutionProvider"]
            si device=="cpu": ["CPUExecutionProvider"]
        Setear threads desde config:
            sess_options.intra_op_num_threads
            sess_options.inter_op_num_threads
        Loggear al cargar: sess.get_providers() y ort.get_available_providers()
    '''
    '''
    3) TFLite: threads + delegate (si esta disponible)
        En tflite_load.py:
        Exponer num_threads desde config (siempre).
        Si device=="gpu":
            intentar crear delegate GPU (si no se puede, loggear y fallback a CPU).
        Loggear al cargar: delegate=GPU/None, num_threads, y si esta usando XNNPACK (si se puede detectar).
    '''
    '''
    4) TensorFlow/Keras: verificacion GPU y control de memoria
        En loader TF:
        Loggear tf.config.list_physical_devices('GPU')
        (Opcional) habilitar memory growth si hay GPU
        Loggear si efectivamente se esta usando GPU en warmup (aunque sea indirecto)
    '''
    '''
    5) Warmup y benchmark correcto (para todos los backends)
        Implementar warmup_runs antes de medir (ej 10–30).
        Medir N iteraciones (ej 200) y calcular:
            promedio FPS
            p95/p99 de t_total (latencia)
        Asegurar que el benchmark:
            usa la misma imagen en loop
            fija el mismo pipeline de pre/post
            evita I/O (no guardar a disco por iteracion)
    '''
    '''
    6) “Guard rails” para no tener auto-engaño con GPU
        Si device=="gpu" pero provider/delegate no se activa:
            levantar WARNING fuerte en logs (porque sino parece que corre en GPU y no)
        Opcion runtime.strict_device=true:
            si pedis GPU y no esta, fallar explicitamente (en vez de fallback silencioso)
    '''
    '''
    7) Perfilado rapido para detectar cuello de botella real
        Reportar por iteracion o por batch: %pre, %inf, %post, %draw.
        Si postprocess domina:
            priorizar NMS/vectorizacion, reducir copias, evitar conversiones innecesarias.
    '''
    '''
    8) Optimizacion “barata” que casi siempre suma FPS (sin GPU todavia)
        Evitar reallocs: reutilizar buffers numpy cuando sea posible.
        Evitar conversiones dtype repetidas (ej uint8→float32 cada frame si no hace falta).
        Minimizar copias CPU (especialmente np.copy, astype sin necesidad).
        Juntar pasos de preprocess en una sola pasada (resize + normalize).
        Si se dibuja boxes:
            que el draw sea opcional en benchmark (porque puede limitar FPS).
    '''
    '''
    9) Validacion final “de verdad”
        Correr benchmark con:
            device=cpu y device=gpu (misma computadora)
            y comparar t_inf y t_total
        Verificar en paralelo:
            nvidia-smi mostrando uso real de GPU
            providers/delegate en logs
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
            self.logger.info("Modelo cargado correctamente.")
            self.logger.info(
                f"""Caracteristicas:
                        Input:
                            Layout: {self.config.InputTensorConfig.layout}
                            dtype:  {self.config.InputTensorConfig.dtype}
                            box format: {self.config.TensorStructure.box_format}
                        Output:
                            unpacker: {self.config.OutputConfig.pack_format}
                """)
        except Exception as e:
            self.logger.exception(f"{e}")



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
        if not isinstance(unpacked, (list, tuple)):
            unpacked = [unpacked]
        adapted_output = [self.output_adapter(list(r)) for r in unpacked]
        iw, ih = self.config.runtime.input_width,  self.config.runtime.input_height
        ow, oh = self.config.runtime.orig_width,   self.config.runtime.orig_height
        md = self.config.runtime.metadata_letter or {}
        self.logger.debug("[DBG] input/orig: input=%dx%d orig=%dx%d", iw, ih, ow, oh)
        self.logger.debug("[DBG] letter: %s", md)
        self.logger.debug(f"[DBG] tensor-space (pre-undo): {np.asarray(adapted_output[:3])}")
        result = self.postprocess_fn(adapted_output)
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
                self.logger.info(
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
