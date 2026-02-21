import numpy as np
import pytest
from api.func.input_pipeline.input_adapter import generate_input_adapter
from api.func.reader_pipeline.config_schema import InputConfig, InputDetection

def test_adapter_rgb_hwc():
    cfg = InputConfig(
        width=1920,
        height=1080,
        input_tensor=InputDetection(layout="HWC", dtype="float32"),
        color_order="RGB",
        channels=3
    )
    adapter = generate_input_adapter(cfg)
    img = np.random.randint(0, 255, (4, 5, 3), dtype=np.uint8)
    out = adapter(img)
    assert out.shape == (4, 5, 3)
    assert out.dtype == np.float32

def test_adapter_bgr_hwc():
    cfg = InputConfig(
        width=1920,
        height=1080,
        input_tensor=InputDetection(layout="HWC", dtype="float32"),
        color_order="BGR",
        channels=3
    )
    adapter = generate_input_adapter(cfg)
    img = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Rojo en RGB
    out = adapter(img)
    # Invertir canales: BGR -> RGB
    assert np.allclose(out[0, 0], [0, 0, 255])

def test_adapter_gray_hwc():
    cfg = InputConfig(
        width=1920,
        height=1080,
        input_tensor=InputDetection(layout="HWC", dtype="float32"),
        color_order="GRAY",
        channels=1
    )
    adapter = generate_input_adapter(cfg)
    img = np.ones((2, 2, 3), dtype=np.uint8) * 255
    out = adapter(img)
    assert out.shape == (2, 2)  # sin canal
    assert out.dtype == np.float32

def test_adapter_invalid_channles():
    cfg = InputConfig(
        width=1920,
        height=1080,
        input_tensor=InputDetection(layout="HWC", dtype="float32"),
        color_order="RGB",
        channels=4
    )
    with pytest.raises(ValueError):
        generate_input_adapter(cfg)

# def test_adapter_invalid_colors():
#     cfg = InputConfig(
#         width=1920,
#         height=1080,
#         input_tensor=InputDetection(layout="HWC", dtype="float32"),
#         color_order="CMYK",     
#         channels=3
#     )
#     with pytest.raises(ValueError):
#         generate_input_adapter(cfg)
#   Este test no tiene sentido porque el error viene de los contratos en config_schema