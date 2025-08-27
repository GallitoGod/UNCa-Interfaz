# tests/test_output_unpacker.py
"""
Tests de unidad para el modulo api.func.output_pipeline.output_unpacker.
Se cubren los formatos: raw, multihead, softmax y efficientdet.
"""

import math
import numpy as np
import pytest
from types import SimpleNamespace

from api.func.output_pipeline.output_unpacker import unpack_out
from api.func.reader_pipeline.config_schema import OutputConfig, OutputTensorConfig


# --------- utilidades ---------

def _mock_output_config(formato: str, threshold: float = 0.5):
    return OutputConfig(
        confidence_threshold=threshold,
        output_tensor=OutputTensorConfig(output_format=formato)
    )

def _assert_lista_de_listas_de_float(x, min_cols: int | None = None):
    assert isinstance(x, list)
    if len(x) == 0:
        return
    for fila in x:
        assert isinstance(fila, list)
        for v in fila:
            assert isinstance(v, (float, int))
            assert not math.isnan(v) and not math.isinf(v)
        if min_cols is not None:
            assert len(fila) >= min_cols

def _assert_fila_deteccion(fila):
    assert isinstance(fila, list)
    assert len(fila) >= 6
    x1, y1, x2, y2, score, cls_id = fila[:6]
    for v in (x1, y1, x2, y2, score, cls_id):
        assert isinstance(v, (float, int))
    assert x1 <= x2 and y1 <= y2
    assert 0.0 <= score <= 1.0


# --------- tests ---------

def test_raw_retorna_input():
    fn = unpack_out(_mock_output_config("raw"))
    dummy = [[1.0, 2.0], [3.0, 4.0]]
    out = fn(dummy)
    assert out == dummy

def test_multihead_concatena():
    fn = unpack_out(_mock_output_config("multihead"))
    arr1 = np.array([[1.0, 2.0]])
    arr2 = np.array([[3.0, 4.0]])
    out = fn([arr1, arr2])
    assert isinstance(out, np.ndarray)
    assert out.shape[-1] == arr1.shape[1] + arr2.shape[1]

def test_softmax_convierte_a_detecciones():
    fn = unpack_out(_mock_output_config("softmax"))
    logits = np.array([[2.0, 1.0, 0.0]])
    out = fn(logits)
    _assert_lista_de_listas_de_float(out, min_cols=6)
    assert out[0][4] == pytest.approx(1.0)  # score
    assert out[0][5] == 0  # class_id esperado

def test_efficientdet_devuelve_detecciones():
    fn = unpack_out(_mock_output_config("efficientdet", threshold=0.3))

    N, C = 3, 4
    boxes = np.array([[
        [0.1, 0.2, 0.5, 0.6],
        [0.2, 0.3, 0.6, 0.7],
        [0.0, 0.0, 0.1, 0.1],
    ]], dtype=np.float32)

    # simulamos que classes contiene distribuciones de probabilidad
    classes = np.array([[
        [0.1, 0.2, 0.7, 0.0],
        [0.05, 0.9, 0.05, 0.0],
        [0.1, 0.1, 0.1, 0.7],
    ]], dtype=np.float32)

    # el unpacker espera tupla (boxes, scores, classes, count)
    raw_output = (boxes, None, classes, N)

    out = fn(raw_output)
    assert isinstance(out, list)
    for fila in out:
        _assert_fila_deteccion(fila)
    # con threshold=0.3 deberían aparecer al menos 2 detecciones
    assert len(out) >= 2

def test_formato_desconocido():
    cfg = OutputConfig(output_tensor=OutputTensorConfig(output_format="cualquiera"))
    with pytest.raises(ValueError):
        unpack_out(cfg)
