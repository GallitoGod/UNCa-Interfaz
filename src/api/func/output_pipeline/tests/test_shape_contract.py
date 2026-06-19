# tests/test_shape_contract.py
# Verifica el contrato de forma UNICO de la capa de unpackers (tarea 2 Fase 2):
# normalize_to_2d centraliza toda la normalizacion que antes estaba duplicada
# entre raw.py y model_controller.inference().
import numpy as np
import pytest

from api.func.output_pipeline.unpackers._shape import normalize_to_2d
from api.func.output_pipeline.unpackers.registry import unpack_out


def test_lista_vacia_es_cero_por_seis():
    out = normalize_to_2d([])
    assert out.shape == (0, 6) and out.dtype == np.float32


def test_lista_de_un_elemento_se_desempaqueta():
    arr = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
    out = normalize_to_2d([arr])
    assert out.shape == (1, 6)
    np.testing.assert_array_equal(out, arr)


def test_lista_de_varios_outputs_es_ambigua():
    with pytest.raises(ValueError):
        normalize_to_2d([np.zeros((1, 6)), np.zeros((1, 6))])


def test_batch_3d_se_aplasta():
    out = normalize_to_2d(np.zeros((1, 5, 6), dtype=np.float32))
    assert out.shape == (5, 6)


def test_fila_1d_se_expande():
    out = normalize_to_2d(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
    assert out.shape == (1, 6)


def test_siempre_float32():
    out = normalize_to_2d(np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64))
    assert out.dtype == np.float32


class _RawCfg:
    pack_format = "raw"


def test_unpack_out_raw_aplica_el_contrato():
    # 'raw' es passthrough; el contrato de forma lo garantiza el wrapper de unpack_out.
    fn = unpack_out(_RawCfg())
    out = fn(np.zeros((1, 7, 12), dtype=np.float64), None)  # batch 3D float64
    assert out.shape == (7, 12) and out.dtype == np.float32


class _PendingCfg:
    def __init__(self, fmt):
        self.pack_format = fmt


@pytest.mark.parametrize("fmt", [
    "softmax_out", "sigmoid_out", "logits_raw",   # clasificacion
    "argmax_map", "softmax_map", "binary_mask",   # segmentacion
])
def test_formatos_cls_seg_son_cables_no_implementados(fmt):
    # CABLE Fase 2 tarea 3: el formato esta reconocido (unpack_out no explota al
    # construir) pero su logica no existe -> NotImplementedError al invocarlo.
    fn = unpack_out(_PendingCfg(fmt))
    with pytest.raises(NotImplementedError):
        fn(np.zeros((1, 4), dtype=np.float32), None)
