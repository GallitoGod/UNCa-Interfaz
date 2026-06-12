import numpy as np
from api.func.input_pipeline.input_transformer import build_letterbox

# build_letterbox ya no toca runtimeShapes: recibe el tamano objetivo por parametro
# y devuelve (imagen, meta). El meta es un dict NUEVO por frame con orig_width/height
# y los parametros del letterbox (scale, pads, letterbox_used).


def test_letterbox_devuelve_meta_del_frame():
    fn = build_letterbox((114, 114, 114), 640, 640)

    img = np.full((480, 640, 3), 255, dtype=np.uint8)
    out, meta = fn(img)

    # La salida debe tener el tamano del tensor objetivo
    assert out.shape == (640, 640, 3)

    # El meta describe el frame original y la transformacion aplicada
    assert meta["orig_width"] == 640
    assert meta["orig_height"] == 480
    assert meta["letterbox_used"] is True
    assert meta["scale"] == 1.0       # 640x480 -> 640x640: no escala (ya entra)
    assert meta["pad_left"] == 0.0
    assert meta["pad_top"] == 80.0    # (640-480)/2 de padding vertical


def test_letterbox_cache_stability_and_change():
    fn = build_letterbox((0, 0, 0), 640, 640)

    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    out1, md1 = fn(img1)

    # Misma resolucion -> mismos parametros (el cache interno no debe alterar nada)
    out2, md2 = fn(img1)

    assert out1.shape == out2.shape == (640, 640, 3)
    assert md1 == md2

    # Cambio de resolucion -> el meta debe recalcularse para el frame nuevo
    img2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    out3, md3 = fn(img2)

    assert out3.shape == (640, 640, 3)
    assert md3 != md2
    assert md3["orig_width"] == 1280
    assert md3["orig_height"] == 720


def test_letterbox_metas_independientes_entre_frames():
    # Clave del diseno sin estado compartido: el meta de un frame NO debe
    # mutar cuando se procesa otro frame (cada llamada devuelve su propio dict).
    fn = build_letterbox((0, 0, 0), 640, 640)

    _, meta_a = fn(np.zeros((480, 640, 3), dtype=np.uint8))
    snapshot_a = dict(meta_a)

    _, meta_b = fn(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert meta_a == snapshot_a          # el meta anterior quedo intacto
    assert meta_b["orig_width"] == 1280  # el nuevo refleja su propio frame
