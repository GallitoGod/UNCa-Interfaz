# tests/test_config_endpoints.py
# Fase 3 — Single Source of Truth de configuraciones:
#   GET  /config/template/{model_type}  -> defaults generados desde Pydantic
#   POST /configs/{name}                -> validacion estricta + escritura segura
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import api.mainAPI as mainAPI
from api.mainAPI import app
from api.func.reader_pipeline.output_shape import introspect_output_shapes

client = TestClient(app)

_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


# --------------------------- GET /config/template ---------------------------

def test_template_detection_trae_defaults_y_anchors():
    r = client.get("/config/template/detection")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] == "detection"
    t = body["template"]
    assert t["model_type"] == "detection"
    assert t["input"]["width"] == 640 and t["input"]["channels"] == 3
    assert t["output"]["pack_format"] == "raw"            # default del schema
    assert t["output"]["confidence_threshold"] == 0.5     # default del schema
    # anchor_defaults solo para detection (familia EfficientDet)
    assert body["anchor_defaults"]["min_level"] == 3
    assert body["anchor_defaults"]["aspect_ratios"] == [1.0, 2.0, 0.5]


def test_template_classification_tiene_num_classes_y_sin_anchors():
    r = client.get("/config/template/classification")
    assert r.status_code == 200
    body = r.json()
    assert body["template"]["output"]["tensor_structure"]["num_classes"] == 1000
    assert "anchor_defaults" not in body


def test_template_segmentation():
    r = client.get("/config/template/segmentation")
    assert r.status_code == 200
    assert r.json()["template"]["output"]["tensor_structure"]["num_classes"] == 21


def test_template_tipo_desconocido_404():
    r = client.get("/config/template/teletransportacion")
    assert r.status_code == 404


# ----------------------------- POST /configs --------------------------------

def test_guardar_config_roundtrip(tmp_path, monkeypatch):
    # La plantilla generada debe round-trip-ear: GET template -> POST -> 200 + archivo.
    monkeypatch.setattr(mainAPI, "CONFIGS_DIR", tmp_path)
    template = client.get("/config/template/detection").json()["template"]

    r = client.post("/configs/mi_modelo", json=template)
    assert r.status_code == 200, r.text

    dest = tmp_path / "mi_modelo.json"
    assert dest.exists()
    saved = json.loads(dest.read_text(encoding="utf-8"))
    assert saved["model_type"] == "detection"


def test_guardar_config_body_invalido_422_no_escribe(tmp_path, monkeypatch):
    monkeypatch.setattr(mainAPI, "CONFIGS_DIR", tmp_path)
    # Falta input/output -> ModelConfig invalida -> 422 y NO se escribe nada.
    r = client.post("/configs/roto", json={"model_type": "detection"})
    assert r.status_code == 422
    assert not (tmp_path / "roto.json").exists()


def test_guardar_config_campo_desconocido_422(tmp_path, monkeypatch):
    # extra="forbid": un campo fantasma debe rechazarse (schema estricto).
    monkeypatch.setattr(mainAPI, "CONFIGS_DIR", tmp_path)
    template = client.get("/config/template/detection").json()["template"]
    template["campo_fantasma"] = 1
    r = client.post("/configs/roto", json=template)
    assert r.status_code == 422
    assert not (tmp_path / "roto.json").exists()


@pytest.mark.parametrize("bad_name", ["with.dot", "espacio malo", "a/b"])
def test_guardar_config_nombre_inseguro_rechazado(tmp_path, monkeypatch, bad_name):
    # Nombres con punto/espacio -> 422 (regex). Con '/' -> 404 (no matchea la ruta).
    # En ningun caso se escribe fuera de configs/.
    monkeypatch.setattr(mainAPI, "CONFIGS_DIR", tmp_path)
    template = client.get("/config/template/detection").json()["template"]
    r = client.post(f"/configs/{bad_name}", json=template)
    assert r.status_code in (404, 422)


# ----------------------- GET /model/output_shape ----------------------------

def test_introspeccion_pytorch_no_disponible_sin_cargar():
    # TorchScript no expone shapes estaticamente -> available False (sin tocar disco).
    out = introspect_output_shapes("/ruta/inventada/modelo.pt")
    assert out["backend"] == "pytorch" and out["available"] is False
    assert out["shapes"] is None


def test_introspeccion_extension_desconocida():
    out = introspect_output_shapes("/ruta/modelo.xyz")
    assert out["available"] is False and out["backend"] == "desconocido"


def test_output_shape_modelo_inexistente_404():
    r = client.get("/model/output_shape/no_existe_este_modelo")
    assert r.status_code == 404


@pytest.mark.skipif(
    not (_MODELS_DIR / "yolov7-tiny.onnx").exists(),
    reason="yolov7-tiny.onnx no presente (git-lfs)")
def test_output_shape_onnx_real():
    r = client.get("/model/output_shape/yolov7-tiny")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "onnxruntime" and body["available"] is True
    assert isinstance(body["shapes"], list) and len(body["shapes"]) >= 1
    # yolov7-tiny: salida tipo [N, 7] (det index + xyxy + conf + cls)
    assert all(isinstance(s, list) for s in body["shapes"])
