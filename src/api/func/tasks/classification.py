# tasks/classification.py — estrategia de CLASIFICACION de imagenes.
# Espeja a tasks/detection.py: posee el armado del pipeline y el loop de inferencia
# especificos del tipo. El ModelController sigue siendo un manager puro: invoca el
# runner que devuelve build_classification_pipeline y mide, sin conocer activaciones,
# top-k ni shapes.

import time
import numpy as np

from ..logger import run_warmup, make_dummy_input
from ..reader_pipeline import Model_loader
from ..input_pipeline import build_preprocessor, generate_input_adapter
from ..output_pipeline import build_classification_postprocessor
from ..output_pipeline.unpackers.registry import unpack_out
from .strategy import TaskStrategy

# Activacion que implica cada pack_format. Sirve solo para detectar configs
# incoherentes al cargar (ver _check_activation_coherence); la activacion real
# la decide el unpacker.
_PACK_FORMAT_ACTIVATION = {
    "softmax_out": "softmax",
    "sigmoid_out": "sigmoid",
    "logits_raw": None,
}


def _check_activation_coherence(output_cfg, logger):
    """
    Avisa (no falla) cuando el JSON se contradice a si mismo.

    El schema tiene TRES campos que hablan de lo mismo: pack_format,
    apply_softmax y apply_sigmoid. La fuente de verdad es pack_format (es la
    clave del registry de unpackers); los otros dos son redundantes y no se
    consumen. Antes que ignorarlos en silencio -que es justo el bug que el
    schema estricto vino a matar- se loguea la contradiccion.

    Son candidatos a poda del schema, igual que se hizo con el flag muerto
    'quantized'.
    """
    pack_format = getattr(output_cfg, "pack_format", "softmax_out")
    implied = _PACK_FORMAT_ACTIVATION.get(pack_format)

    declared = None
    if getattr(output_cfg, "apply_softmax", False):
        declared = "softmax"
    if getattr(output_cfg, "apply_sigmoid", False):
        declared = "sigmoid" if declared is None else "softmax+sigmoid"

    if declared is not None and declared != implied:
        logger.warning(
            "Config incoherente: pack_format '%s' implica activacion %s pero "
            "apply_softmax/apply_sigmoid declaran '%s'. Manda pack_format; los "
            "flags apply_* no se usan.",
            pack_format, implied, declared,
        )

    ts = getattr(output_cfg, "tensor_structure", None)
    if ts is not None and getattr(ts, "output_format", None) == "probabilities" and implied is not None:
        logger.info(
            "output_format='probabilities': el modelo ya emite probabilidades, "
            "asi que NO se aplica %s. (pack_format '%s' solo nombra la activacion "
            "que corresponderia si la salida fueran logits.)",
            implied, pack_format,
        )


def build_classification_pipeline(config, model_path, logger):
    """
    Arma el pipeline completo de clasificacion y devuelve un 'runner' autocontenido.

    runner(img, debug=False) -> (result, timings):
      - result : ndarray (K, 2) [class_id, score], ordenado por score desc.
      - timings: dict {pre_ms, inf_ms, post_ms} para el PerfMeter del controller.

    A diferencia de deteccion NO hay output_adapter: no hay columnas de caja que
    reordenar. El vector de clases sale del unpacker ya en el unico orden posible
    (indice = class_id).
    """
    predict_fn = Model_loader.load(model_path, config.runtime, logger)
    preprocess_fn = build_preprocessor(config.input, config.runtime)
    input_adapter = generate_input_adapter(config.input)

    # Warmup opcional: las primeras inferencias suelen ser lentas (alocacion/JIT).
    w = config.runtime.warmup
    if w.enabled and w.runs > 0:
        dummy_input = make_dummy_input(preprocess_fn, input_adapter, config.input)
        run_warmup(predict_fn, dummy_input, runs=w.runs, logger=logger)

    _check_activation_coherence(config.output, logger)

    unpack_fn = unpack_out(config.output)
    postprocess_fn = build_classification_postprocessor(config.output)

    # Contrato declarado en el JSON: cuantas clases dice tener el modelo. Se valida
    # contra el tensor real en cada frame (es una comparacion de enteros, gratis) y
    # falla ruidosamente si no coincide: es el analogo del chequeo de indices de
    # tensor_structure que hace deteccion.
    declared_classes = int(config.output.tensor_structure.num_classes)

    def run(img, debug=False):
        # 1. preprocess. El meta (tamano original + letterbox) se descarta: la
        #    clasificacion no devuelve coordenadas, no hay nada que deshacer.
        t_pre0 = time.perf_counter()
        pre, _frame_meta = preprocess_fn(img)
        adapted_input = input_adapter(pre)
        t_pre1 = time.perf_counter()

        # 2. inferencia del backend
        t_inf0 = time.perf_counter()
        raw_output = predict_fn(adapted_input)
        t_inf1 = time.perf_counter()

        # 3. desempaquetado -> vector (C,) de puntajes por clase
        t_post0 = time.perf_counter()
        scores = unpack_fn(raw_output, getattr(config, "runtime", None))

        if scores.shape[0] != declared_classes:
            raise ValueError(
                f"El JSON declara num_classes={declared_classes} pero el modelo "
                f"devolvio {scores.shape[0]} clases. Corregir "
                "'output.tensor_structure.num_classes' en el config."
            )

        # 4. postprocess: umbral (en vivo) + top-k + orden
        result = postprocess_fn(scores)
        t_post1 = time.perf_counter()

        if debug:
            top = [(int(c), round(float(s), 4)) for c, s in result[:3]]
            logger.debug(
                "Clasificacion: %d clases sobre el umbral. Top: %s", len(result), top)

        timings = {
            "pre_ms": (t_pre1 - t_pre0) * 1000,
            "inf_ms": (t_inf1 - t_inf0) * 1000,
            "post_ms": (t_post1 - t_post0) * 1000,
        }
        return result, timings

    return run


def serialize_classification(result):
    """
    Serializa el resultado de dominio (ndarray (K,2)) al formato del envelope:
        [{"cls": <int>, "score": <float>}, ...]

    4 decimales (no 2 como deteccion): son probabilidades, y en multi-etiqueta las
    clases secundarias viven en el rango 0.00x, donde redondear a 2 las volveria 0.
    """
    return [{"cls": int(cls), "score": round(float(score), 4)} for cls, score in result]


# Estrategia exportada: la consume el registry.
classification_strategy = TaskStrategy(
    task="classification",
    build_pipeline=build_classification_pipeline,
    serialize=serialize_classification,
)
