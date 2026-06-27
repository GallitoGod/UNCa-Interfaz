# tasks/segmentation.py — estrategia de SEGMENTACION (registrada, sin implementar).
# Misma idea que classification: presente en el registry para despacho uniforme, pero
# build_pipeline levanta TaskNotImplemented (501) hasta que se implemente el decode de
# mascara + serializador, sin tocar el WS ni el controller.

from .strategy import TaskStrategy
from .errors import TaskNotImplemented

# Contrato de salida ACORDADO (spec 2026-06-27 §4), pendiente de implementar:
#   result = {"mask": "<rle|b64>", "shape": [h, w]}


def build_segmentation_pipeline(config, model_path, logger):
    raise TaskNotImplemented(
        "model_type 'segmentation' reconocido pero su pipeline todavia no esta "
        "implementado (decode de mascara + serializador pendientes)."
    )


def serialize_segmentation(result):
    # Nunca alcanzado: el 501 corta en la carga antes de llegar a inferir/serializar.
    raise TaskNotImplemented("serializador de 'segmentation' no implementado.")


segmentation_strategy = TaskStrategy(
    task="segmentation",
    build_pipeline=build_segmentation_pipeline,
    serialize=serialize_segmentation,
)
