# test_yolo_v8_unpacker.py — el head "Detect" de Ultralytics v8+ (2026-08-27).
#
# Cubre las tres diferencias con yolo_flat que motivaron un unpacker aparte —el tensor
# transpuesto, la ausencia de objectness y la conversion a xyxy vectorizada— y los dos
# bugs que aparecieron al integrarlo, que son los casos que mas vale la pena blindar
# porque NINGUNO de los dos lanza una excepcion: producen cajas silenciosamente mal.

import numpy as np
import pytest

from api.func.output_pipeline.unpackers.registry import UNPACKERS_REGISTRY, unpack_out
from api.func.output_pipeline.unpackers.yolo_v8 import build_yolo_v8
from api.func.reader_pipeline.config_schema import DetectionOutput
from api.func.tasks.detection import _NEEDS_ADAPTER


N_CLASES = 10          # VisDrone, el modelo que motivo esto
N_ANCLAS = 8400        # candidatos de un head 640x640


def _cfg(num_classes=N_CLASES):
    return DetectionOutput.model_validate({
        "pack_format": "yolo_v8",
        "tensor_structure": {
            "box_format": "xyxy",
            "coordinates": {"x1": 0, "y1": 1, "x2": 2, "y2": 3},
            "confidence_index": 4,
            "class_index": 5,
            "num_classes": num_classes,
        },
    })


def _tensor(n_anclas=4, num_classes=N_CLASES):
    """Tensor crudo (1, 4+C, N) con cajas y scores reconocibles."""
    t = np.zeros((1, 4 + num_classes, n_anclas), dtype=np.float32)
    for i in range(n_anclas):
        t[0, 0, i] = 100 + i * 10      # cx
        t[0, 1, i] = 200 + i * 10      # cy
        t[0, 2, i] = 20                # w
        t[0, 3, i] = 40                # h
        t[0, 4 + (i % num_classes), i] = 0.9   # una clase distinta gana en cada ancla
    return t


# ── Los dos bugs que se comieron la integracion ─────────────────────────────

def test_desenvuelve_la_lista_del_predict_fn():
    """
    BUG 1. session.run() de ONNX devuelve SIEMPRE una lista, aunque el grafo tenga una
    sola salida, y cada unpacker la desenvuelve por su cuenta (el runner solo normaliza
    la SALIDA del unpacker, no su entrada). Sin esto el array queda (1,1,F,N), cae en
    'ndim != 2' y el unpacker devuelve CERO detecciones sin quejarse: el modelo parece
    no detectar nada.
    """
    fn = build_yolo_v8(_cfg())
    envuelto = fn([_tensor()], None)       # como lo entrega el loader
    pelado = fn(_tensor(), None)
    assert len(envuelto) == 4
    assert np.array_equal(envuelto, pelado)


def test_endereza_el_tensor_transpuesto():
    """
    BUG 2 (el que motivo el archivo). El head llega (1, 4+C, N): una FILA por
    caracteristica. Si se leyera como (N, F) saldrian 4+C 'detecciones' de basura — 14
    en el modelo real — sin ninguna excepcion.
    """
    fn = build_yolo_v8(_cfg())
    salida = fn(_tensor(n_anclas=N_ANCLAS), None)
    assert salida.shape == (N_ANCLAS, 6), (
        "deben salir tantas filas como ANCLAS, no como caracteristicas")


# ── Orientacion ─────────────────────────────────────────────────────────────

def test_usa_num_classes_para_decidir_la_orientacion():
    """Con num_classes declarado la decision es exacta, no una heuristica."""
    fn = build_yolo_v8(_cfg())
    # Un tensor CUADRADO (14x14) donde la heuristica de "el eje chico son features"
    # no puede decidir: solo num_classes desambigua.
    t = np.zeros((1, 4 + N_CLASES, 4 + N_CLASES), dtype=np.float32)
    t[0, 0, :] = 50.0
    assert fn(t, None).shape == (4 + N_CLASES, 6)


def test_sin_num_classes_cae_a_la_heuristica():
    """Las caracteristicas son siempre muchas menos que los candidatos (14 vs 8400)."""
    fn = build_yolo_v8(_cfg(num_classes=None))
    assert fn(_tensor(n_anclas=N_ANCLAS), None).shape == (N_ANCLAS, 6)


def test_acepta_el_tensor_ya_orientado():
    """(N, 4+C) tiene que pasar sin transponer."""
    fn = build_yolo_v8(_cfg())
    ya = _tensor(n_anclas=50)[0].T          # (50, 14)
    assert fn(ya, None).shape == (50, 6)


# ── Semantica: sin objectness, y cxcywh -> xyxy ─────────────────────────────

def test_el_score_es_el_maximo_de_clase_sin_objectness():
    """
    v5/v7 traen [cx,cy,w,h,obj,p0..pC] y el score es obj*max(p). v8 elimino esa
    columna. Multiplicar por una quinta columna inexistente tomaria la probabilidad
    de la clase 0 como si fuera objectness.
    """
    fn = build_yolo_v8(_cfg(num_classes=3))
    t = np.zeros((1, 7, 1), dtype=np.float32)
    t[0, :4, 0] = [100, 100, 10, 10]
    t[0, 4, 0] = 0.10      # clase 0 (seria la 'objectness' si nos confundieramos)
    t[0, 5, 0] = 0.80      # clase 1, la ganadora
    t[0, 6, 0] = 0.30
    fila = fn(t, None)[0]
    assert fila[4] == pytest.approx(0.80), "el score debe ser max(clase), no un producto"
    assert int(fila[5]) == 1


def test_convierte_cxcywh_a_xyxy():
    """Sale en el formato estandar: por eso NO usa output_adapter."""
    fn = build_yolo_v8(_cfg(num_classes=1))
    t = np.zeros((1, 5, 1), dtype=np.float32)
    t[0, :4, 0] = [100, 200, 20, 40]     # cx, cy, w, h
    t[0, 4, 0] = 0.5
    x1, y1, x2, y2 = fn(t, None)[0][:4]
    assert (x1, y1, x2, y2) == pytest.approx((90.0, 180.0, 110.0, 220.0))


def test_las_coordenadas_pasan_sin_escalar():
    """
    El head emite pixeles del tensor de entrada. Escalar aca romperia el contrato:
    el JSON declara out_coords_space y del resto se ocupa el postprocesador.
    """
    fn = build_yolo_v8(_cfg(num_classes=1))
    t = np.zeros((1, 5, 1), dtype=np.float32)
    t[0, :4, 0] = [637.0, 300.0, 2.0, 2.0]
    assert fn(t, None)[0][0] == pytest.approx(636.0)


# ── Bordes ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("entrada", [
    [],                                          # lista vacia
    np.empty((1, 14, 0), dtype=np.float32),      # sin candidatos
    np.zeros((1, 3, 5), dtype=np.float32),       # menos de 4+1 caracteristicas
])
def test_bordes_devuelven_matriz_vacia_bien_formada(entrada):
    """Nunca revienta el frame: devuelve (0,6), que es lo que el runner espera."""
    salida = build_yolo_v8(_cfg())(entrada, None)
    assert salida.shape == (0, 6) and salida.dtype == np.float32


def test_varias_salidas_es_un_error_explicito():
    """Un modelo 'seg'/'pose' emite dos tensores: mejor fallar que adivinar."""
    with pytest.raises(ValueError, match="2 outputs"):
        build_yolo_v8(_cfg())([_tensor(), _tensor()], None)


# ── Cableado ────────────────────────────────────────────────────────────────

def test_esta_registrado_y_el_schema_lo_acepta():
    assert "yolo_v8" in UNPACKERS_REGISTRY
    assert callable(unpack_out(_cfg()))


def test_no_usa_output_adapter():
    """
    Sale ya en formato estandar. Ademas de correcto (el orden de columnas del head es
    fijo, no configurable) es lo que evita 8400 llamadas Python por frame: medido,
    22,2 ms contra 0,13 vectorizado.
    """
    assert "yolo_v8" not in _NEEDS_ADAPTER


def test_el_schema_rechaza_un_pack_format_inventado():
    """Guarda de la convencion: agregar un unpacker exige tocar tambien el Literal."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        DetectionOutput.model_validate({"pack_format": "yolo_v99"})
