# test_output_adapter.py — el adapter vectorizado (2026-08-27).
#
# El adapter paso de llamarse UNA VEZ POR FILA a recibir el tensor entero. La ganancia
# es grande donde el head es crudo (efficientdet-lite0: postproceso de 37,8 a 1,4 ms),
# pero el riesgo es el mismo en todos los modelos, porque este codigo esta en el camino
# de TODA deteccion. Por eso el test central no comprueba "da algo razonable" sino que
# da EXACTAMENTE lo mismo que la implementacion anterior, contra una referencia
# fila-a-fila escrita aca abajo a proposito.

import numpy as np
import pytest

from api.func.output_pipeline.output_adapter import (
    _generate_box_converter,
    generate_output_adapter,
)
from api.func.reader_pipeline.config_schema import TensorDetection


# ── La implementacion VIEJA, conservada como referencia ─────────────────────
# Copia literal de la version fila-a-fila que existio hasta el 2026-08-27. No se toca:
# su unico trabajo es discrepar si alguien cambia la semantica del adapter sin querer.

def _referencia_fila_a_fila(ts: TensorDetection):
    fmt = ts.box_format or "xyxy"
    coords = ts.coordinates or {"x1": 0, "y1": 1, "x2": 2, "y2": 3}

    if fmt == "xyxy" or fmt == "yxyx":
        def caja(row):
            return [row[coords["x1"]], row[coords["y1"]],
                    row[coords["x2"]], row[coords["y2"]]]
    elif fmt == "cxcywh":
        def caja(row):
            return [row[coords["cx"]] - row[coords["w"]] / 2,
                    row[coords["cy"]] - row[coords["h"]] / 2,
                    row[coords["cx"]] + row[coords["w"]] / 2,
                    row[coords["cy"]] + row[coords["h"]] / 2]
    else:
        raise ValueError(fmt)

    def fn(rows):
        return [[*caja(r), r[ts.confidence_index], r[ts.class_index]] for r in rows]
    return fn


def _ts(**kw):
    base = {"box_format": "xyxy", "coordinates": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            "confidence_index": 6, "class_index": 5}
    base.update(kw)
    return TensorDetection.model_validate(base)


# Las tres configuraciones REALES del repo, mas la de yolo_flat.
CONFIGS = {
    "yolov7-tiny (raw)": _ts(),
    "efficientdet (anchor_deltas, yxyx)": _ts(
        box_format="yxyx", coordinates={"y1": 0, "x1": 1, "y2": 2, "x2": 3},
        confidence_index=4, class_index=5),
    "yolo_flat (cxcywh)": _ts(
        box_format="cxcywh", coordinates={"cx": 0, "cy": 1, "w": 2, "h": 3},
        confidence_index=4, class_index=5),
    "best (yolo_v8, xyxy directo)": _ts(
        box_format="xyxy", coordinates={"x1": 0, "y1": 1, "x2": 2, "y2": 3},
        confidence_index=4, class_index=5),
}


@pytest.mark.parametrize("nombre", list(CONFIGS))
def test_identico_a_la_implementacion_fila_a_fila(nombre):
    """
    EL test de esta refactorizacion: misma salida, bit a bit, en las cuatro
    configuraciones de tensor que existen en el repo.
    """
    ts = CONFIGS[nombre]
    rng = np.random.default_rng(7)
    # 7 columnas cubren el indice mas alto de todas las configs (yolov7 usa hasta el 6).
    filas = (rng.random((500, 7), dtype=np.float32) * 640).astype(np.float32)

    nuevo = generate_output_adapter(ts)(filas)
    viejo = np.asarray(_referencia_fila_a_fila(ts)(filas), dtype=np.float32)

    assert nuevo.shape == viejo.shape
    assert np.array_equal(nuevo, viejo), (
        f"{nombre}: el adapter vectorizado difiere de la referencia fila-a-fila")


def test_yxyx_y_xyxy_comparten_implementacion_a_proposito():
    """
    No es un descuido heredado: el reordenamiento lo hace el dict 'coordinates' del
    JSON (un modelo yxyx declara {y1:0, x1:1, y2:2, x2:3}), asi que ambos formatos se
    leen por NOMBRE de clave y la conversion resulta la misma. Eran dos ramas identicas
    en la version anterior y siguen siendolo.
    """
    coords = {"y1": 0, "x1": 1, "y2": 2, "x2": 3}
    filas = np.array([[10, 20, 30, 40, 0.9, 2]], dtype=np.float32)
    como_yxyx = generate_output_adapter(
        _ts(box_format="yxyx", coordinates=coords, confidence_index=4, class_index=5))(filas)
    como_xyxy = generate_output_adapter(
        _ts(box_format="xyxy", coordinates=coords, confidence_index=4, class_index=5))(filas)
    assert np.array_equal(como_yxyx, como_xyxy)
    # y lee por nombre: x1 esta en la columna 1, no en la 0
    assert list(como_yxyx[0][:4]) == [20.0, 10.0, 40.0, 30.0]


def test_cxcywh_convierte_a_esquinas():
    ts = _ts(box_format="cxcywh", coordinates={"cx": 0, "cy": 1, "w": 2, "h": 3},
             confidence_index=4, class_index=5)
    filas = np.array([[100, 200, 20, 40, 0.5, 3]], dtype=np.float32)
    assert list(generate_output_adapter(ts)(filas)[0]) == pytest.approx(
        [90.0, 180.0, 110.0, 220.0, 0.5, 3.0])


def test_devuelve_matriz_bien_formada_sin_filas():
    """El runner espera (N,6) siempre: un frame sin candidatos no es un caso especial."""
    salida = generate_output_adapter(_ts())(np.empty((0, 7), dtype=np.float32))
    assert salida.shape == (0, 6) and salida.dtype == np.float32


def test_formato_desconocido_falla_al_ARMAR_no_por_frame():
    """
    La validacion cae al construir el pipeline (una vez, al cargar), no dentro del hot
    path. Un box_format invalido tiene que romper la carga del modelo, no el frame 900.
    """
    ts = _ts()
    object.__setattr__(ts, "box_format", "inventado")   # el schema no lo permitiria
    with pytest.raises(ValueError, match="Formato desconocido"):
        generate_output_adapter(ts)


def test_es_una_sola_llamada_para_todo_el_tensor():
    """
    Guarda del contrato nuevo: si alguien volviera a llamarlo fila a fila, cada 'fila'
    seria un vector 1D y esto fallaria en vez de devolver algo raro en silencio.
    """
    adapter = generate_output_adapter(_ts())
    filas = np.zeros((4, 7), dtype=np.float32)
    assert adapter(filas).shape == (4, 6)
    # una fila suelta (1D) no es un tensor de detecciones: devuelve vacio, no basura
    assert adapter(filas[0]).shape == (0, 6)


# ── El conversor de cajas, probado directo ──────────────────────────────────
# Los tres casos que ya cubria este archivo antes de la vectorizacion, portados al
# contrato nuevo: _generate_box_converter recibe la matriz (N,K) y devuelve cuatro
# COLUMNAS, no cuatro escalares de una fila.

def test_convert_cxcywh_a_xyxy():
    """cx=50, cy=40, w=20, h=10  =>  x1=40, y1=35, x2=60, y2=45"""
    convert = _generate_box_converter("cxcywh", {"cx": 0, "cy": 1, "w": 2, "h": 3})
    filas = np.array([[50.0, 40.0, 20.0, 10.0, 0.9, 3.0]], dtype=np.float32)
    x1, y1, x2, y2 = convert(filas)
    assert np.isclose([x1[0], y1[0], x2[0], y2[0]], [40.0, 35.0, 60.0, 45.0], atol=1e-6).all()


def test_convert_yxyx_a_xyxy():
    """y1,x1,y2,x2 -> xyxy: el reordenamiento lo hace el dict 'coordinates'."""
    convert = _generate_box_converter("yxyx", {"y1": 0, "x1": 1, "y2": 2, "x2": 3})
    filas = np.array([[10.0, 20.0, 30.0, 40.0, 0.5, 1.0]], dtype=np.float32)
    x1, y1, x2, y2 = convert(filas)
    assert (x1[0], y1[0], x2[0], y2[0]) == (20.0, 10.0, 40.0, 30.0)


def test_convert_xywh_a_xyxy():
    """x=15, y=5, w=10, h=20 => x1=15, y1=5, x2=25, y2=25"""
    convert = _generate_box_converter("xywh", {"x": 0, "y": 1, "w": 2, "h": 3})
    filas = np.array([[15.0, 5.0, 10.0, 20.0, 0.6, 2.0]], dtype=np.float32)
    x1, y1, x2, y2 = convert(filas)
    assert (x1[0], y1[0], x2[0], y2[0]) == (15.0, 5.0, 25.0, 25.0)


def test_convert_trabaja_sobre_columnas_enteras():
    """
    Lo que cambio en la vectorizacion: el conversor ve TODAS las filas de una vez.
    Antes se lo llamaba una vez por caja candidata desde el runner.
    """
    convert = _generate_box_converter("cxcywh", {"cx": 0, "cy": 1, "w": 2, "h": 3})
    filas = np.tile(np.array([[50.0, 40.0, 20.0, 10.0]], dtype=np.float32), (100, 1))
    x1, _, x2, _ = convert(filas)
    assert x1.shape == (100,) and x2.shape == (100,)
    assert np.all(x1 == 40.0) and np.all(x2 == 60.0)
