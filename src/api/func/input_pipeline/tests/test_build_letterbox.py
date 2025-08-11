import numpy as np
from api.func.input_pipeline.input_transformer import build_letterbox


def test_letterbox_padding_and_scale():
    # Imagen original simulada
    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    letterbox_fn, info = build_letterbox(1280, 720, pad_color=(0, 0, 0))

    out = letterbox_fn(img)
    assert out.shape == (720, 1280, 3)

    # Chequeo de proporciones
    expected_scale = min(1280 / 1920, 720 / 1080)
    assert np.isclose(info["scale"], expected_scale)

    # Padding deberia existir si no es escalado exacto
    assert info["pad_left"] >= 0
    assert info["pad_top"] >= 0
    assert info["letterbox_used"] is True

