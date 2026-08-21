# Tests del seam por model_type: registry, serializador de deteccion y excepciones tipadas.
import numpy as np
import pytest

from api.func.tasks.registry import get_strategy, TASK_STRATEGIES
from api.func.tasks.errors import UnknownModelType, TaskNotImplemented
from api.func.tasks.detection import serialize_detection, detection_strategy
from api.func.tasks.classification import serialize_classification, classification_strategy


# ── Registry ──────────────────────────────────────────────────────────────────

def test_get_strategy_detection():
    assert get_strategy("detection") is detection_strategy
    assert get_strategy("detection").task == "detection"


def test_get_strategy_unknown_raises_typed():
    with pytest.raises(UnknownModelType):
        get_strategy("no_existe")


def test_registry_tiene_los_tres_tipos():
    assert set(TASK_STRATEGIES) == {"detection", "classification", "segmentation"}


def test_get_strategy_classification():
    assert get_strategy("classification") is classification_strategy
    assert get_strategy("classification").task == "classification"


# ── Build de SEG: 501 honesto via TaskNotImplemented ──────────────────────────
# Clasificacion se implemento el 2026-08-13 y salio de esta lista; solo queda
# segmentacion sin pipeline (falta el decode de mascara).

@pytest.mark.parametrize("model_type", ["segmentation"])
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


# ── Serializador de clasificacion ─────────────────────────────────────────────

def test_serialize_classification_formato_del_envelope():
    # (K,2) [class_id, score] -> [{"cls": int, "score": float}]
    arr = np.array([[663.0, 0.61523], [813.0, 0.43897]], dtype=np.float32)
    out = serialize_classification(arr)
    assert out == [{"cls": 663, "score": 0.6152}, {"cls": 813, "score": 0.439}]
    # cls debe salir como int de Python, no como float ni np.float32
    assert isinstance(out[0]["cls"], int)


def test_serialize_classification_redondea_a_4_decimales():
    # 2 decimales (como deteccion) aplastaria a 0.0 las clases secundarias de un
    # multi-etiqueta, que viven en el rango 0.00x.
    arr = np.array([[7.0, 0.0043210]], dtype=np.float32)
    assert serialize_classification(arr) == [{"cls": 7, "score": 0.0043}]


def test_serialize_classification_matriz_vacia():
    arr = np.empty((0, 2), dtype=np.float32)
    assert serialize_classification(arr) == []
