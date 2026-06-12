# model_controller.py
import os
import time
import threading
import numpy as np
from .logger import setup_model_logger, PerfMeter, run_warmup, make_dummy_input
from .reader_pipeline import Model_loader
from .reader_pipeline import load_model_config
from .input_pipeline import build_preprocessor, generate_input_adapter
from .output_pipeline import buildPostprocessor, generate_output_adapter
from .output_pipeline.unpackers.registry import unpack_out
from .output_pipeline.unpackers.anchor_gen import generate_efficientdet_anchors

'''
    Tiene que comportarce como un controlador del backend dependiente de los eventos del cliente.
    Debe ser capaz de:
        1_ Cargar modelos
        2_ Configurar propiedades
        3_ Ejecutar en CPU, o en otro procesador
        4_ Liberar recursos

    Patron aplicado: Strategy.
'''

# boxes_scores ya entrega [x1,y1,x2,y2,conf,cls] en formato estandar del sistema.
# Aplicar el adapter encima reordena mal las coords (swapea x/y de vuelta a yxyx).
# raw y yolo_flat si necesitan el adapter porque salen en el espacio del tensor sin reordenar.
_NEEDS_ADAPTER = {"raw", "yolo_flat", "tflite_detpost", "anchor_deltas"}


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

    def __init__(self):
        # RLock: validate_pipeline() llama a inference() con el lock ya tomado.
        self._lock = threading.RLock()
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
        """
        Carga el modelo y arma el pipeline completo.
        Atomico: el estado del controller solo se actualiza si TODO el armado salio bien.
        Si algo falla, el controller queda descargado y la excepcion SE PROPAGA para que
        la API pueda responder con el error real (antes respondia "ok" con un modelo roto).
        """
        logger = setup_model_logger(os.path.basename(model_path).split(".")[0])
        with self._lock:
            try:
                config = load_model_config(model_path)

                if config.model_type != "detection":
                    raise NotImplementedError(
                        f"model_type '{config.model_type}' todavia no esta soportado: "
                        "el pipeline actual solo cubre deteccion de objetos."
                    )

                predict_fn = Model_loader.load(model_path, config.runtime, logger)
                preprocess_fn = build_preprocessor(config.input, config.runtime)
                input_adapter = generate_input_adapter(config.input)

                w = config.runtime.warmup
                if w.enabled and w.runs > 0:
                    dummy_input = make_dummy_input(preprocess_fn, input_adapter, config.input)
                    run_warmup(predict_fn, dummy_input, runs=w.runs, logger=logger)

                if config.output.pack_format == "anchor_deltas":
                    ac = config.output.anchor_config
                    if ac is None:
                        raise ValueError(
                            "pack_format 'anchor_deltas' requiere 'anchor_config' en output "
                            "para poder generar la tabla de anchors.")
                    rs = config.runtime.runtimeShapes
                    rs.anchors = generate_efficientdet_anchors(
                        config.input.height, config.input.width, ac)
                    rs.box_variance = list(ac.box_variance)
                    logger.info(f"Anchors generados: {rs.anchors.shape[0]} "
                                f"(niveles {ac.min_level}-{ac.max_level}, "
                                f"{ac.num_scales} escalas x {len(ac.aspect_ratios)} aspects)")

                unpack_fn = unpack_out(config.output)
                output_adapter = generate_output_adapter(config.output.tensor_structure)
                postprocess_fn = buildPostprocessor(config.output, config.runtime)
            except Exception:
                logger.exception(f"Fallo la carga de '{model_path}'. El controller queda descargado.")
                self.unload_model()
                raise

            # Commit atomico del nuevo pipeline
            self.perf.reset()
            self._frame_idx = 0
            self.model_format = os.path.splitext(model_path)[1].lower()
            self.config = config
            self.predict_fn = predict_fn
            self.preprocess_fn = preprocess_fn
            self.input_adapter = input_adapter
            self.unpack_fn = unpack_fn
            self.output_adapter = output_adapter
            self.postprocess_fn = postprocess_fn
            self.logger = logger

            logger.info("Modelo cargado correctamente.")
            logger.debug(
                f"""Caracteristicas:
                Input:
                    Layout: {config.input.input_str.layout if config.input.input_str else 'HWC'}
                    dtype:  {config.input.input_str.dtype if config.input.input_str else 'float32'}
                Output:
                    box format: {config.output.tensor_structure.box_format}
                    unpacker: {config.output.pack_format}
                """)

    def validate_pipeline(self) -> dict:
        """
        Validacion cruzada JSON <-> modelo: corre una inferencia end-to-end sobre un
        frame dummy para detectar al cargar (y no en pleno stream) que el contrato
        declarado en el JSON coincide con lo que el modelo realmente devuelve.
        """
        with self._lock:
            if self.predict_fn is None:
                raise RuntimeError("No hay modelo cargado para validar.")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            try:
                result = self.inference(dummy)
            except Exception as e:
                raise ValueError(
                    "Validacion post-carga fallida: el modelo no produce la salida que "
                    f"el JSON declara. Detalle: {e}"
                ) from e
            finally:
                # La corrida dummy no debe contaminar metricas ni contadores
                self.perf.reset()
                self._frame_idx = 0
            self.logger.info(
                f"Validacion post-carga OK ({len(result)} detecciones sobre frame dummy).")
            return {"dummy_detections": len(result)}

    def inference(self, img):
        with self._lock:
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

            pack_fmt = (getattr(self.config.output, "pack_format", "raw") or "raw").lower()
            if pack_fmt in _NEEDS_ADAPTER:
                if unpacked.shape[0] > 0:
                    # Validacion: los indices declarados en el JSON deben caber en el tensor real
                    ts = self.config.output.tensor_structure
                    max_idx = max([*ts.coordinates.values(), ts.confidence_index, ts.class_index])
                    if max_idx >= unpacked.shape[1]:
                        raise ValueError(
                            f"tensor_structure declara indices hasta {max_idx} pero el tensor "
                            f"desempaquetado tiene {unpacked.shape[1]} columnas. Revisar "
                            "'coordinates'/'confidence_index'/'class_index' en el JSON.")
                adapted_output = [self.output_adapter(r) for r in unpacked]
            else:
                adapted_output = unpacked  # ya en [x1,y1,x2,y2,conf,cls]

            log_this_frame = (self._frame_idx % self._log_every == 0)
            if log_this_frame:
                rs = self.config.runtime.runtimeShapes
                self.logger.debug("[DBG] input/orig: input=%dx%d orig=%dx%d letter=%s",
                                  rs.input_width, rs.input_height,
                                  rs.orig_width, rs.orig_height, rs.metadata_letter or {})

            result = self.postprocess_fn(adapted_output)
            if log_this_frame:
                self.logger.debug("Inferencia ejecutada: %d detecciones. Primeras: %s",
                                  len(result), result[:3])
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
        """Valida y aplica el umbral. Lanza si no hay modelo o el valor esta fuera de rango."""
        if self.config is None:
            raise RuntimeError("No hay modelo cargado: no se puede actualizar el umbral.")
        if not (0.0 <= new_threshold <= 1.0):
            raise ValueError(f"Umbral de confianza fuera de rango [0, 1]: {new_threshold}")
        # El postprocesador lee este campo en cada llamada -> efecto inmediato ("en vivo")
        self.config.output.confidence_threshold = float(new_threshold)
        if self.logger:
            self.logger.info(f"Confianza actualizada a {new_threshold}.")

    def unload_model(self):
        with self._lock:
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
2_ Preprocesar frame--------------------------------------(hecho) <-- input_transformer.py
3_ Adaptar el preproceso generico a la IA especifica------(hecho) <-- input_adapter.py
4_ Generar la inferencia----------------------------------(hecho) <-- model_loader.py
5_ Adaptar raw_output al formato generico del controlador-(hecho) <-- output_adapter.py
6_ Postprocesar detections--------------------------------(hecho) <-- output_transformer.py
7_ Devolver al cliente------------------------------------(hecho) <-- mainAPI.py
"""
