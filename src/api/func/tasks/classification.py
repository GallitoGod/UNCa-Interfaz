# tasks/classification.py — estrategia de CLASIFICACION (registrada, sin implementar).
# Existe en el registry desde el dia uno para que el despacho sea uniforme, pero su
# pipeline todavia no esta implementado: build_pipeline levanta TaskNotImplemented
# (la API lo traduce a 501 honesto). Cuando se implemente, este archivo se completa
# (unpacker softmax/sigmoid + top-k + serializador) sin tocar el WS ni el controller.

from .strategy import TaskStrategy
from .errors import TaskNotImplemented

# Contrato de salida ACORDADO (spec 2026-06-27 §4), pendiente de implementar:
#   result = [{"cls": <int>, "score": <float>}, ...]  (top-k de probabilidades)


def build_classification_pipeline(config, model_path, logger):
    raise TaskNotImplemented(
        "model_type 'classification' reconocido pero su pipeline todavia no esta "
        "implementado (unpacker + top-k + serializador pendientes)."
    )


def serialize_classification(result):
    # Nunca alcanzado: el 501 corta en la carga antes de llegar a inferir/serializar.
    raise TaskNotImplemented("serializador de 'classification' no implementado.")


classification_strategy = TaskStrategy(
    task="classification",
    build_pipeline=build_classification_pipeline,
    serialize=serialize_classification,
)
