# tasks/registry.py — despacho por model_type.
# Reemplaza al viejo _PIPELINE_BUILDERS del controller: una sola fuente de verdad
# que mapea model_type -> TaskStrategy. Agregar un tipo nuevo es registrar su
# estrategia aca (un archivo nuevo en tasks/), sin tocar el controller ni el WS.

from .strategy import TaskStrategy
from .errors import UnknownModelType
from .detection import detection_strategy
from .classification import classification_strategy
from .segmentation import segmentation_strategy

TASK_STRATEGIES = {
    s.task: s
    for s in (detection_strategy, classification_strategy, segmentation_strategy)
}


def get_strategy(model_type: str) -> TaskStrategy:
    """Devuelve la TaskStrategy del model_type. Levanta UnknownModelType si no existe."""
    try:
        return TASK_STRATEGIES[model_type]
    except KeyError:
        raise UnknownModelType(
            f"model_type '{model_type}' no reconocido. "
            f"Validos: {sorted(TASK_STRATEGIES)}"
        ) from None
