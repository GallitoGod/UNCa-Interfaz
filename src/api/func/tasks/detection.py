# tasks/detection.py — estrategia de DETECCION de objetos.
# Posee el armado del pipeline y el loop de inferencia detection-especificos, que
# antes vivian en el ModelController. El controller ahora solo invoca el runner que
# build_detection_pipeline devuelve, sin conocer adapters, shapes ni indices.

import time
import numpy as np

from ..logger import run_warmup, make_dummy_input
from ..reader_pipeline import Model_loader
from ..input_pipeline import build_preprocessor, generate_input_adapter
from ..output_pipeline import buildPostprocessor, generate_output_adapter
from ..output_pipeline.unpackers.registry import unpack_out
from ..output_pipeline.unpackers.anchor_gen import generate_efficientdet_anchors
from .strategy import TaskStrategy

# boxes_scores ya entrega [x1,y1,x2,y2,conf,cls] en formato estandar del sistema.
# Aplicar el adapter encima reordena mal las coords (swapea x/y de vuelta a yxyx).
# raw y yolo_flat si necesitan el adapter porque salen en el espacio del tensor sin reordenar.
_NEEDS_ADAPTER = {"raw", "yolo_flat", "tflite_detpost", "anchor_deltas"}


def build_detection_pipeline(config, model_path, logger):
    """
    Arma el pipeline completo de deteccion y devuelve un 'runner' autocontenido.

    runner(img, debug=False) -> (result, timings):
      - result : ndarray (N,6) [x1,y1,x2,y2,conf,cls] en px de la imagen original.
      - timings: dict {pre_ms, inf_ms, post_ms} para alimentar el PerfMeter del controller.

    Todo el conocimiento detection-especifico (decision del adapter, normalizacion de
    shape, validacion de indices del tensor_structure) vive aca.
    """
    predict_fn = Model_loader.load(model_path, config.runtime, logger)
    preprocess_fn = build_preprocessor(config.input, config.runtime)
    input_adapter = generate_input_adapter(config.input)

    # Warmup opcional: primeras inferencias suelen ser lentas (alocacion/JIT).
    w = config.runtime.warmup
    if w.enabled and w.runs > 0:
        dummy_input = make_dummy_input(preprocess_fn, input_adapter, config.input)
        run_warmup(predict_fn, dummy_input, runs=w.runs, logger=logger)

    # anchor_deltas (EfficientDet/SSD crudos): la tabla de anchors NO viaja en el JSON,
    # se genera al cargar a partir de output.anchor_config y se deja en runtimeShapes.
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

    pack_fmt = (getattr(config.output, "pack_format", "raw") or "raw").lower()
    needs_adapter = pack_fmt in _NEEDS_ADAPTER

    def run(img, debug=False):
        # 1. preprocess -> (tensor, meta). El meta (orig size + letterbox) viaja con
        #    el frame hasta el post; cada inferencia es autocontenida (reforma 8).
        t_pre0 = time.perf_counter()
        pre, frame_meta = preprocess_fn(img)
        adapted_input = input_adapter(pre)
        t_pre1 = time.perf_counter()

        # 2. inferencia del backend
        t_inf0 = time.perf_counter()
        raw_output = predict_fn(adapted_input)
        t_inf1 = time.perf_counter()

        # 3. desempaquetado + normalizacion de shape -> matriz 2D (N,K)
        t_post0 = time.perf_counter()
        unpacked = unpack_fn(raw_output, getattr(config, "runtime", None))

        if isinstance(unpacked, (list, tuple)):
            if len(unpacked) == 0:
                unpacked = np.empty((0, 6), dtype=np.float32)
            elif len(unpacked) == 1 and hasattr(unpacked[0], "ndim"):
                unpacked = unpacked[0]
            else:
                raise ValueError(
                    f"unpack_fn devolvio {len(unpacked)} outputs; normalizacion ambigua.")

        unpacked = np.asarray(unpacked)
        if unpacked.ndim == 3 and unpacked.shape[0] == 1:
            unpacked = unpacked[0]
        if unpacked.ndim == 1:
            unpacked = unpacked[None, :]

        # 4. adapter SOLO si el pack_format lo necesita (boxes_scores ya viene estandar)
        if needs_adapter:
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

        if debug:
            # input_width/height son constantes de carga (runtimeShapes);
            # el tamano original y el letterbox son del frame actual (meta).
            rs = config.runtime.runtimeShapes
            logger.debug("[DBG] input/orig: input=%dx%d orig=%dx%d letter=%s",
                         rs.input_width, rs.input_height,
                         frame_meta.get("orig_width", 0),
                         frame_meta.get("orig_height", 0),
                         frame_meta)

        # 5. postprocess: conf filter + top-k + NMS + undo letterbox (usa el meta)
        result = postprocess_fn(adapted_output, frame_meta)
        if debug:
            logger.debug("Inferencia ejecutada: %d detecciones. Primeras: %s",
                         len(result), result[:3])
        t_post1 = time.perf_counter()

        timings = {
            "pre_ms": (t_pre1 - t_pre0) * 1000,
            "inf_ms": (t_inf1 - t_inf0) * 1000,
            "post_ms": (t_post1 - t_post0) * 1000,
        }
        return result, timings

    return run


def serialize_detection(result):
    """
    Serializa el resultado de dominio (ndarray (N,6)) al formato del envelope:
    lista de [x1,y1,x2,y2,conf,cls] redondeados a 2 decimales.
    No traga: si 'result' no es iterable de filas numericas, propaga.
    """
    return [[round(float(v), 2) for v in det] for det in result]


# Estrategia exportada: la consume el registry.
detection_strategy = TaskStrategy(
    task="detection",
    build_pipeline=build_detection_pipeline,
    serialize=serialize_detection,
)
