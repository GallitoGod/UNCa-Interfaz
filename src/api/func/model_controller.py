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
    1) Config JSON: runtime controlado por modelo (NO hardcode)     <--- COMPLETADO
        Agregar en schema/config algo asi (idea, no literal):
        runtime.device: "cpu" | "gpu".
        runtime.backend: "onnxruntime" | "tflite" | "tensorflow" (si aplica).
        runtime.threads: intra_op, inter_op, num_threads.
        runtime.onnx.providers: lista ordenada (ej ["CUDAExecutionProvider","CPUExecutionProvider"]).
        runtime.tflite.delegates: ["gpu"] o vacio.
        runtime.warmup_runs.
    '''
    '''
    2) ONNX Runtime: GPU habilitable y fallback     <--- COMPLETADO
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
    3) TFLite: threads + delegate (si esta disponible)      <--- COMPLETADO
        En tflite_load.py:
        Exponer num_threads desde config (siempre).
        Si device=="gpu":
            intentar crear delegate GPU (si no se puede, loggear y fallback a CPU).
        Loggear al cargar: delegate=GPU/None, num_threads, y si esta usando XNNPACK (si se puede detectar).
    '''
    '''
    4) TensorFlow/Keras: verificacion GPU y control de memoria      <--- COMPLETADO
        En loader TF:
        Loggear tf.config.list_physical_devices('GPU')
        (Opcional) habilitar memory growth si hay GPU
        Loggear si efectivamente se esta usando GPU (aunque sea indirecto)
    '''
    '''
    PARA TENER EN CUENTA: Optimizacion “barata” que casi siempre suma FPS (sin GPU todavia)
        Evitar reallocs: reutilizar buffers numpy cuando sea posible.
        Evitar conversiones dtype repetidas (ej uint8→float32 cada frame si no hace falta).
        Minimizar copias CPU (especialmente np.copy, astype sin necesidad).
        Juntar pasos de preprocess en una sola pasada (resize + normalize).
        Si se dibuja boxes:
            que el draw sea opcional en benchmark (porque puede limitar FPS).
    '''
    '''
    PARA TENER EN CUENTA: Validacion final “de verdad”
        Correr benchmark con:
            device=cpu y device=gpu (misma computadora)
            y comparar t_inf y t_total
        Verificar en paralelo:
            nvidia-smi mostrando uso real de GPU
            providers/delegate en logs
    '''

    '''
        DEBUG - [DBG] input/orig: input=640x640 orig=1920x1080
        DEBUG - [DBG] letter: {'scale': 0.3333333333333333, 'pad_left': 0.0, 'pad_top': 140.0, 'letterbox_used': True}
        DEBUG - [DBG] tensor-space (pre-undo): [[4.1021619e+02 1.7992230e+02 6.3928540e+02 4.9856915e+02 8.2283282e-01
        1.7000000e+01]
        [3.6798859e-01 1.6471231e+02 2.0640732e+02 4.6118118e+02 7.9173315e-01
        1.7000000e+01]
        [2.8473242e+02 1.9341101e+02 4.3834485e+02 4.9710425e+02 7.5424951e-01
        1.7000000e+01]]
        DEBUG - [[1230.6485595703125, 119.76690673828125, 1917.856201171875, 1075.7073974609375, 0.8228328227996826, 17.0], [1.1039657592773438, 74.13693237304688, 619.221923828125, 963.5435180664062, 0.7917331457138062, 17.0], [854.197265625, 160.2330322265625, 1315.0345458984375, 1071.312744140625, 0.7542495131492615, 17.0], [379.1082763671875, 73.42080688476562, 888.6564331054688, 1068.0205078125, 0.6822399497032166, 17.0], [1.78564453125, 643.9934692382812, 426.9784240722656, 1069.9405517578125, 0.4531806409358978, 17.0]]
        INFO - Inferencia ejecutada: 5 detecciones
        Tiempo de carga total: 181.01 ms
        Tiempo (wall) por 1 imagen: 54.04 ms
        Total 200 imgs: 11.638 s | FPS (wall): 17.18
        PerfMeter | n=211 avg=57.31ms p95=68.93ms fps=17.45 | pre=14.28 inf=41.90 post=1.13

        BIEN, se puede ver que el postprocesado esta perfecto, el numero parece error blanco.
        Tambien es importante ver el preprocesado, todavia puedo hacer mucho ahi, pero sin lugar a dudas
        el problema se encuentra en las inferencias, no es mio, el modelo esta siendo ejecutado en un 
        Ryzen 5 5600X, llega lo jodido: como hago que todo funcione en CUDA ????
        
        Pensaba que eso ya lo tenia resuelto pero por alguna razon el sistema no lee CUDAExecutionProvider
        solo lee CPUExecutionProvider, si paso la inferencia a gpu la performance explota, mas que nada porque
        correria en una GPU RTX 3060.
    '''

    '''
        INFO - Inferencia ejecutada: 28 detecciones
        DEBUG - [DBG] input/orig: input=320x320 orig=1920x1080
        DEBUG - [DBG] letter: {'scale': 0.16666666666666666, 'pad_left': 0.0, 'pad_top': 70.0, 'letterbox_used': True}
        DEBUG - [DBG] tensor-space (pre-undo): [[-0.03878175  0.06809887 -1.0533311  -0.7728281   0.00197633  0.        ]
        [ 0.19823961 -0.01707495 -1.803082   -1.4310396   0.00244074  0.        ]
        [-0.0435522   0.36151633 -1.5297735  -0.7218313   0.00407261  0.        ]]
        DEBUG - [[0.0, 0.0, 1.480668067932129, 0.0, 0.741368293762207, 18.0], [0.27678394317626953, 0.0, 1.798208475112915, 0.0, 0.7221113443374634, 18.0], [0.0, 0.0, 0.0, 0.0, 0.7220731973648071, 18.0], [0.22902309894561768, 0.0, 0.46020984649658203, 0.0, 0.7190607786178589, 18.0], [0.0, 0.0, 0.0, 0.0, 0.7027531862258911, 18.0]]
        INFO - Inferencia ejecutada: 28 detecciones
        DEBUG - [DBG] input/orig: input=320x320 orig=1920x1080
        DEBUG - [DBG] letter: {'scale': 0.16666666666666666, 'pad_left': 0.0, 'pad_top': 70.0, 'letterbox_used': True}
        DEBUG - [DBG] tensor-space (pre-undo): [[-0.03878175  0.06809887 -1.0533311  -0.7728281   0.00197633  0.        ]
        [ 0.19823961 -0.01707495 -1.803082   -1.4310396   0.00244074  0.        ]
        [-0.0435522   0.36151633 -1.5297735  -0.7218313   0.00407261  0.        ]]
        DEBUG - [[0.0, 0.0, 1.480668067932129, 0.0, 0.741368293762207, 18.0], [0.27678394317626953, 0.0, 1.798208475112915, 0.0, 0.7221113443374634, 18.0], [0.0, 0.0, 0.0, 0.0, 0.7220731973648071, 18.0], [0.22902309894561768, 0.0, 0.46020984649658203, 0.0, 0.7190607786178589, 18.0], [0.0, 0.0, 0.0, 0.0, 0.7027531862258911, 18.0]]
        INFO - Inferencia ejecutada: 28 detecciones
        Tiempo de carga total: 201.38 ms
        Tiempo (wall) por 1 imagen: 66.55 ms
        Total 200 imgs: 15.682 s | FPS (wall): 12.75
        PerfMeter | n=211 avg=76.93ms p95=128.21ms fps=13.00 | pre=4.66 inf=42.62 post=29.64

        Bueno... Es curioso que para este modelo el preprocesado sea simple y el postprocesado complejo,
        me doy una idea del porque: el preprocesado es mas barato porque la entrada es NHWC float32 y el 
        postprocesado es caro por boxes_scores + undo letterbox/escala, probablemente deberia volver a ver 
        la implementacion del undo letterbox (loops o copias merman el rendimiento).
        Inferencia sigue siendo grande, pero ahora post también es un cuello importante.

        Aparte se ve un problema en las escalas de pixeles de los boxes.
        El output parece estar mal interpretado (coords negativas / >1), asi que hay un bug de mapeo/espacio de coordenadas.
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

        adapted_output = [self.output_adapter(r) for r in unpacked]

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
