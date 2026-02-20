import numpy as np
from api.func.input_pipeline.input_transformer import build_letterbox
from api.func.reader_pipeline.config_schema import RuntimeConfig, RuntimeShapes

def test_letterbox_updates_runtime_shapes():
    runtime = RuntimeConfig(runtimeShapes=RuntimeShapes(input_width=640, input_height=640))
    fn = build_letterbox((114,114,114), runtime)

    img = np.full((480, 640, 3), 255, dtype=np.uint8)
    out = fn(img)

    assert out.shape == (640, 640, 3)

    shapes = runtime.runtimeShapes
    assert shapes.orig_width == 640
    assert shapes.orig_height == 480

    md = shapes.metadata_letter
    assert md["letterbox_used"] is True
    assert md["scale"] == 1.0
    assert md["pad_left"] == 0.0
    assert md["pad_top"] == 80.0


def test_letterbox_cache_stability_and_change():
    runtime = RuntimeConfig(runtimeShapes=RuntimeShapes(input_width=640, input_height=640))
    fn = build_letterbox((0,0,0), runtime)

    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    out1 = fn(img1)
    md1 = dict(runtime.runtimeShapes.metadata_letter)

    # Misma resolucion -> metadata igual
    out2 = fn(img1)
    md2 = dict(runtime.runtimeShapes.metadata_letter)

    assert out1.shape == out2.shape == (640, 640, 3)
    assert md1 == md2

    # Cambio de resolucion -> metadata deberia cambiar
    img2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    out3 = fn(img2)
    md3 = dict(runtime.runtimeShapes.metadata_letter)

    assert out3.shape == (640, 640, 3)
    assert md3 != md2
    assert runtime.runtimeShapes.orig_width == 1280
    assert runtime.runtimeShapes.orig_height == 720