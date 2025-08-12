import json
import os
import pytest
from pydantic import ValidationError

from api.func.reader_pipeline.json_reader import load_model_config
from api.func.reader_pipeline.config_schema import ModelConfig


def test_load_model_config_valido(tmp_path):
    model_file = tmp_path / "modeloFalso.onnx"
    model_file.write_text("fake content")

    config_data = {
        "model_type": "classification",
        "input": {
            "width": 224,
            "height": 224,
            "channels": 3,
            "normalize": True,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "scale": True,
            "letterbox": False,
            "auto_pad_color": [114, 114, 114],
            "preserve_aspect_ratio": True,
            "color_order": "RGB",
            "input_tensor": {
                "layout": "HWC",
                "dtype": "float32",
                "quantized": False
            }
        },
        "output": {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.45,
            "apply_nms": False,
            "tensor_structure": None,
            "output_tensor": {
                "output_format": "raw"
            }
        }
    }

    config_file = tmp_path / "configs" / "modeloFalso.json"
    config_file.parent.mkdir()
    config_file.write_text(json.dumps(config_data))

    old_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        cfg = load_model_config(str(model_file))
        assert cfg.model_type == "classification"
        assert cfg.input.width == 224
    finally:
        os.chdir(old_cwd)



def test_load_model_config_no_encontrado(tmp_path):
    """Debe lanzar FileNotFoundError si el archivo de configuracion no existe."""
    model_file = tmp_path / "modeloFalso.onnx"
    model_file.write_text("fake content")

    # No se crea el archivo en configs/
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(FileNotFoundError):
            load_model_config(str(model_file))
    finally:
        os.chdir(old_cwd)


def test_load_model_config_json_invalido(tmp_path):
    """Debe lanzar ValueError si el JSON es invalido."""
    model_file = tmp_path / "modeloFalso.onnx"
    model_file.write_text("fake content")

    config_file = tmp_path / "configs" / "modeloFalso.json"
    config_file.parent.mkdir()
    config_file.write_text("{invalid_json}")  # JSON mal formado

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValueError):
            load_model_config(str(model_file))
    finally:
        os.chdir(old_cwd)


def test_load_model_config_datos_invalidos(tmp_path):
    model_file = tmp_path / "modeloFalso.onnx"
    model_file.write_text("fake content")

    # Falta campo requerido "model_type"
    config_data = {
        "input": {
            "width": 224,
            "height": 224,
            "channels": 3,
            "normalize": True,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "scale": True,
            "letterbox": False,
            "auto_pad_color": [114, 114, 114],
            "preserve_aspect_ratio": True,
            "color_order": "RGB",
            "input_tensor": {
                "layout": "HWC",
                "dtype": "float32",
                "quantized": False
            }
        },
        "output": {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.45,
            "apply_nms": False,
            "tensor_structure": None,
            "output_tensor": {
                "output_format": "raw"
            }
        }
    }
    config_file = tmp_path / "configs" / "modeloFalso.json"
    config_file.parent.mkdir()
    config_file.write_text(json.dumps(config_data))

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValidationError):
            load_model_config(str(model_file))
    finally:
        os.chdir(old_cwd)
