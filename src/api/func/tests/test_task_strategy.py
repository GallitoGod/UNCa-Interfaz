# Tests del seam por model_type: registry, serializador de deteccion y excepciones tipadas.
import numpy as np
import pytest

from api.func.tasks.registry import get_strategy, TASK_STRATEGIES
from api.func.tasks.errors import UnknownModelType, TaskNotImplemented
from api.func.tasks.detection import serialize_detection, detection_strategy


# ── Registry ──────────────────────────────────────────────────────────────────

def test_get_strategy_detection():
    assert get_strategy("detection") is detection_strategy
    assert get_strategy("detection").task == "detection"


def test_get_strategy_unknown_raises_typed():
    with pytest.raises(UnknownModelType):
        get_strategy("no_existe")


def test_registry_tiene_los_tres_tipos():
    assert set(TASK_STRATEGIES) == {"detection", "classification", "segmentation"}


# ── Build de CLS/SEG: 501 honesto via TaskNotImplemented ──────────────────────

@pytest.mark.parametrize("model_type", ["classification", "segmentation"])
def test_build_no_implementado_levanta_typed(model_type):
    strategy = get_strategy(model_type)
    with pytest.raises(TaskNotImplemented):
        strategy.build_pipeline(config=None, model_path="x", logger=None)


# ── Mapeo HTTP garantizado por subclaseo ──────────────────────────────────────

def test_excepciones_son_subclases_para_mapeo_http():
    # _load_and_validate mapea ValueError->422 y NotImplementedError->501;
    # el subclaseo garantiza que el mapeo siga funcionando sin tocar la frontera.
    assert issubclass(UnknownModelType, ValueError)
    assert issubclass(TaskNotImplemented, NotImplementedError)


# ── Serializador de deteccion ─────────────────────────────────────────────────

def test_serialize_detection_redondea_a_2_decimales():
    arr = np.array([[10.123, 20.456, 30.0, 40.0, 0.987, 1.0]], dtype=np.float32)
    out = serialize_detection(arr)
    assert out == [[10.12, 20.46, 30.0, 40.0, 0.99, 1.0]]


def test_serialize_detection_matriz_vacia():
    arr = np.empty((0, 6), dtype=np.float32)
    assert serialize_detection(arr) == []
