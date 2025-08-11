import numpy as np
import pytest
from api.func.input_pipeline.input_adapter import generate_layout_converter


def test_layout_chw():
    img = np.random.rand(4, 5, 3)  # HWC
    converter = generate_layout_converter("CHW")
    out = converter(img)
    assert out.shape == (3, 4, 5)  # Canales primero


def test_layout_nhwc():
    img = np.random.rand(4, 5, 3)  # HWC
    converter = generate_layout_converter("NHWC")
    out = converter(img)
    assert out.shape == (1, 4, 5, 3)  # Batch primero


def test_layout_nchw():
    img = np.random.rand(4, 5, 3)  # HWC
    converter = generate_layout_converter("NCHW")
    out = converter(img)
    assert out.shape == (1, 3, 4, 5)


def test_layout_hwc():
    img = np.random.rand(4, 5, 3)  # HWC
    converter = generate_layout_converter("HWC")
    out = converter(img)
    assert out.shape == img.shape  # Sin cambios


def test_layout_invalido():
    with pytest.raises(ValueError):
        generate_layout_converter("XYZ")(np.zeros((4, 4, 3)))


