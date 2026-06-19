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
    Tiene que comportarce como un ADMINISTRADOR de pipelines dependiente de los eventos
    del cliente. Debe ser capaz de:
        1_ Cargar modelos (despachando al pipeline segun model_type)
        2_ Configurar propiedades
        3_ Ejecutar en CPU, o en otro procesador
        4_ Liberar recursos

    Patron aplicado: Strategy. El controller NO conoce los detalles de cada pipeline:
    lee el model_type del JSON y delega el armado al builder correspondiente
    (_PIPELINE_BUILDERS). Hoy 'detection' esta implementado; 'classification' y
    'segmentation' estan reconocidas por el despachador pero su pipeline todavia no
    existe (levantan NotImplementedError -> la API responde 501 honesto).
'''

# boxes_scores ya entrega [x1,y1,x2,y2,conf,cls] en formato estandar del sistema.
# Aplicar el adapter encima reordena mal las coords (swapea x/y de vuelta a yxyx).
# raw y yolo_flat si necesitan el adapter porque salen en el espacio del tensor sin reordenar.
_NEEDS_ADAPTER = {"raw", "yolo_flat", "tflite_detpost", "anchor_deltas"}


# ===========================================================================
# Builders de pipeline por tipo de modelo (despachador — tarea 3 Fase 2)
# ---------------------------------------------------------------------------
# Cada builder recibe (model_path, config, logger) y devuelve un dict con las
# funciones del pipeline que el controller commitea de forma atomica. Si un tipo
# todavia no esta soportado, su builder levanta NotImplementedError (la API lo
# mapea a 501). Asi se reemplaza el viejo "if model_type != 'detection': raise"
# por un despacho data-driven y extensible.
# ===========================================================================

def _build_detection_pipeline(model_path: str, config, logger) -> dict:
    """Arma el pipeline de DETECCION de objetos (el unico implementado hoy)."""
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

    # unpack_out ya envuelve el contrato de forma (N,K) float32 2D (tarea 2):
    # el controller no normaliza shapes, confia en esta capa.
    unpack_fn = unpack_out(config.output)
    output_adapter = generate_output_adapter(config.output.tensor_structure)
    postprocess_fn = buildPostprocessor(config.output, config.runtime)

    return {
        "predict_fn": predict_fn,
        "preprocess_fn": preprocess_fn,
        "input_adapter": input_adapter,
        "unpack_fn": unpack_fn,
        "output_adapter": output_adapter,
        "postprocess_fn": postprocess_fn,
    }


def _build_classification_pipeline(model_path: str, config, logger) -> dict:
    """CABLE Fase 2 tarea 3 — clasificacion. El despachador ya enruta hasta aca; la
    LOGICA todavia no existe. El lado de INPUT del pipeline es generico y ya esta
    disponible (Model_loader.load / build_preprocessor / generate_input_adapter):
    cuando se conecte, esos tres se reutilizan tal cual. Falta conectar:

      1) UNPACKER: implementar el builder real para pack_format softmax_out /
         sigmoid_out / logits_raw en unpackers/registry.py (hoy stub en
         _PENDING_FORMATS). Debe entregar un vector (num_classes,) de scores.
      2) POSTPROCESO: top-k (ClassificationOutput.top_k) + confidence_threshold +
         mapear ids con label_map -> [(class_id, score), ...] ordenado por score.
      3) CONTRATO CON EL CLIENTE (decision de diseno PENDIENTE): hoy /video_stream
         devuelve cajas [x1,y1,x2,y2,conf,cls]. Definir el JSON de clasificacion
         (ej: {"classification": [{"id": .., "score": ..}]}) y su render en overlay.js.
      4) INFERENCE: el lado de salida de inference() es detection-shaped (adapter +
         postprocess que devuelve cajas). Para CLS habra que despachar el postproceso
         por tipo (o que cada pipeline traiga su propia inference_fn).

    Se requieren varios modelos de CLS para implementarlo y validarlo en serio."""
    raise NotImplementedError(
        "model_type 'classification' reconocido por el despachador pero su pipeline "
        "todavia no esta implementado (unpacker + postproceso top-k + contrato de "
        "salida con el cliente pendientes). Ver el checklist en _build_classification_pipeline.")


def _build_segmentation_pipeline(model_path: str, config, logger) -> dict:
    """CABLE Fase 2 tarea 3 — segmentacion semantica. Mismo patron que clasificacion:
    el INPUT generico ya existe; falta conectar la LOGICA. Falta:

      1) UNPACKER: builder real para pack_format argmax_map / softmax_map /
         binary_mask en unpackers/registry.py (hoy stub en _PENDING_FORMATS).
      2) POSTPROCESO: decodificar la mascara (argmax por pixel o umbral sobre
         softmax_map), aplicar output_stride / resize_to_input al tamano original
         usando el meta del frame, y opcional colormap por clase.
      3) CONTRATO CON EL CLIENTE (decision de diseno PENDIENTE): una mascara HxW es
         pesada para el WS actual (cajas JSON). Decidir formato (PNG/RLE/array) y el
         render en el canvas. <-- es trabajo de backend Y frontend (encaja con Fase 4).
      4) INFERENCE: igual que CLS, el lado de salida hoy es detection-shaped.

    Se requieren varios modelos de SEG para implementarlo y validarlo en serio."""
    raise NotImplementedError(
        "model_type 'segmentation' reconocido por el despachador pero su pipeline "
        "todavia no esta implementado (decodificacion de mascara + contrato de salida "
        "con el cliente pendientes). Ver el checklist en _build_segmentation_pipeline.")


_PIPELINE_BUILDERS = {
    "detection": _build_detection_pipeline,
    "classification": _build_classification_pipeline,
    "segmentation": _build_segmentation_pipeline,
}


class ModelController:

    # TODO [INSTANCIA]: Segmentacion de instancias (ej: YOLOv8-seg, Mask R-CNN).
    #   A diferencia de la segmentacion semantica, combina deteccion de objetos + mascara por instancia.
    #   Requiere correr dos postprocesados en cadena:
    #     1_ Postprocesado de deteccion: obtener boxes + coeficientes de mascara por deteccion.
    #     2_ Postprocesado de mascara: combinar coeficientes con prototipos (YOLOv8-seg) o recortar mascaras
    #        directamente por ROI (Mask R-CNN). Binarizar con mask_threshold.
    #   Campos adicionales en el schema: mask_threshold (float), mask_channels (int, ej: 32 en YOLOv8-seg).
    #   El despachador deberia tener un builder "instance_segmentation".

    def __init__(self):
        # _lock: serializa SOLO carga/descarga y el snapshot atomico del pipeline.
        #   La inferencia NO lo sostiene durante el trabajo pesado (pre/predict/post),
        #   por eso varios frames pueden inferir en paralelo (tarea 1 Fase 2). Los
        #   predict_fn de cada backend son thread-safe (ORT/TF/Torch lo son nativamente;
        #   TFLite se serializa con su propio lock interno en el loader).
        self._lock = threading.RLock()
        # _stats_lock: protege el estado mutable de diagnostico (perf + contador de
        #   frames) que varios hilos de inferencia tocan a la vez.
        self._stats_lock = threading.Lock()
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
        Carga el modelo y arma el pipeline completo despachando por model_type.
        Atomico: el estado del controller solo se actualiza si TODO el armado salio bien.
        Si algo falla, el controller queda descargado y la excepcion SE PROPAGA para que
        la API pueda responder con el error real (antes respondia "ok" con un modelo roto).
        """
        logger = setup_model_logger(os.path.basename(model_path).split(".")[0])
        with self._lock:
            try:
                config = load_model_config(model_path)

                builder = _PIPELINE_BUILDERS.get(config.model_type)
                if builder is None:
                    raise NotImplementedError(
                        f"model_type '{config.model_type}' no reconocido. "
                        f"Tipos validos: {sorted(_PIPELINE_BUILDERS)}.")

                # Despacho al pipeline correspondiente (detection / classification / ...).
                pipeline = builder(model_path, config, logger)
            except Exception:
                logger.exception(f"Fallo la carga de '{model_path}'. El controller queda descargado.")
                self.unload_model()
                raise

            # Commit atomico del nuevo pipeline (bajo _lock: o se ve todo el set
            # nuevo, o el anterior; nunca un estado a medias).
            self.perf.reset()
            self._frame_idx = 0
            self.model_format = os.path.splitext(model_path)[1].lower()
            self.config = config
            self.predict_fn = pipeline["predict_fn"]
            self.preprocess_fn = pipeline["preprocess_fn"]
            self.input_adapter = pipeline["input_adapter"]
            self.unpack_fn = pipeline["unpack_fn"]
            self.output_adapter = pipeline.get("output_adapter")
            self.postprocess_fn = pipeline["postprocess_fn"]
            self.logger = logger

            logger.info("Modelo cargado correctamente.")
            logger.debug(
                f"""Caracteristicas:
                Tipo: {config.model_type}
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

        NO sostiene _lock mientras corre la inferencia (se apoya en el snapshot
        atomico de inference()), para no reintroducir la serializacion que la tarea 1
        justamente elimina.
        """
        with self._lock:
            loaded = self.predict_fn is not None
            logger = self.logger
        if not loaded:
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
            with self._stats_lock:
                self._frame_idx = 0
        logger.info(
            f"Validacion post-carga OK ({len(result)} detecciones sobre frame dummy).")
        return {"dummy_detections": len(result)}

    def inference(self, img):
        # 1) Snapshot atomico del pipeline bajo lock CORTO. El lock NO se sostiene
        #    durante pre/predict/post: asi varios frames infieren en paralelo.
        with self._lock:
            preprocess_fn = self.preprocess_fn
            input_adapter = self.input_adapter
            predict_fn = self.predict_fn
            unpack_fn = self.unpack_fn
            output_adapter = self.output_adapter
            postprocess_fn = self.postprocess_fn
            config = self.config
            logger = self.logger

        if predict_fn is None or config is None:
            raise RuntimeError("No hay modelo cargado: no se puede inferir.")

        t0 = time.perf_counter()

        t_pre0 = time.perf_counter()
        # El preprocesador devuelve (tensor, meta): el meta lleva el tamano
        # original del frame y los parametros del letterbox. Viaja junto al
        # frame hasta el postprocesador en vez de vivir en estado compartido,
        # asi cada inferencia es autocontenida (y concurrente).
        pre, frame_meta = preprocess_fn(img)
        adapted_input = input_adapter(pre)
        t_pre1 = time.perf_counter()

        t_inf0 = time.perf_counter()
        raw_output = predict_fn(adapted_input)
        t_inf1 = time.perf_counter()

        t_post0 = time.perf_counter()

        # Contrato de la capa de unpackers (tarea 2): unpack_fn SIEMPRE devuelve un
        # ndarray 2D float32 (N, K). El controller NO normaliza shapes: confia en el.
        unpacked = unpack_fn(raw_output, getattr(config, "runtime", None))

        pack_fmt = (getattr(config.output, "pack_format", "raw") or "raw").lower()
        if pack_fmt in _NEEDS_ADAPTER:
            if unpacked.shape[0] > 0:
                # Validacion: los indices declarados en el JSON deben caber en el tensor real
                ts = config.output.tensor_structure
                max_idx = max([*ts.coordinates.values(), ts.confidence_index, ts.class_index])
                if max_idx >= unpacked.shape[1]:
                    raise ValueError(
                        f"tensor_structure declara indices hasta {max_idx} pero el tensor "
                        f"desempaquetado tiene {unpacked.shape[1]} columnas. Revisar "
                        "'coordinates'/'confidence_index'/'class_index' en el JSON.")
            adapted_output = [output_adapter(r) for r in unpacked]
        else:
            adapted_output = unpacked  # ya en [x1,y1,x2,y2,conf,cls]

        # El postprocesador necesita el meta para devolver las cajas al
        # espacio de la imagen original (undo letterbox / re-escala).
        result = postprocess_fn(adapted_output, frame_meta)
        t_post1 = time.perf_counter()

        t1 = time.perf_counter()

        pre_ms   = (t_pre1 - t_pre0) * 1000
        inf_ms   = (t_inf1 - t_inf0) * 1000
        post_ms  = (t_post1 - t_post0) * 1000
        total_ms = (t1 - t0) * 1000

        # 2) Estado de diagnostico compartido: bajo _stats_lock (varios hilos).
        with self._stats_lock:
            self.perf.push(pre_ms, inf_ms, post_ms, total_ms)
            self._frame_idx += 1
            idx = self._frame_idx

        if idx % self._log_every == 0:
            rs = config.runtime.runtimeShapes
            logger.debug("[DBG] input/orig: input=%dx%d orig=%dx%d letter=%s",
                         rs.input_width, rs.input_height,
                         frame_meta.get("orig_width", 0),
                         frame_meta.get("orig_height", 0),
                         frame_meta)
            s = self.perf.stats()
            if s:
                logger.debug(
                    "PERF n=%d avg=%.2fms p95=%.2fms fps=%.2f | pre=%.2f inf=%.2f post=%.2f",
                    s["n"], s["avg_ms"], s["p95_ms"], s["fps_avg"],
                    s["pre_avg_ms"], s["inf_avg_ms"], s["post_avg_ms"],
                )
            logger.debug("Inferencia ejecutada: %d detecciones. Primeras: %s",
                         len(result), result[:3])
        return result

    def update_confidence(self, new_threshold: float):
        """Valida y aplica el umbral. Lanza si no hay modelo o el valor esta fuera de rango."""
        with self._lock:
            config = self.config
            logger = self.logger
        if config is None:
            raise RuntimeError("No hay modelo cargado: no se puede actualizar el umbral.")
        if not (0.0 <= new_threshold <= 1.0):
            raise ValueError(f"Umbral de confianza fuera de rango [0, 1]: {new_threshold}")
        # El postprocesador lee este campo en cada llamada -> efecto inmediato ("en vivo").
        # Escritura de un solo float: atomica en CPython, no necesita el lock pesado.
        config.output.confidence_threshold = float(new_threshold)
        if logger:
            logger.info(f"Confianza actualizada a {new_threshold}.")

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
