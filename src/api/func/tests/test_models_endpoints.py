# test_models_endpoints.py — tests de los endpoints HTTP que reemplazaron al IPC de
# disco del frontend (thin client sin disco): GET /models, GET/POST /configs/{name},
# POST /models/upload, y los templates GET /config/template/{type}.
#
# Los endpoints usan los globals MODELS_DIR / CONFIGS_DIR del modulo mainAPI; se los
# redirige a directorios temporales con monkeypatch para no tocar el repo real.

import json
import pytest
from fastapi.testclient import TestClient

import api.mainAPI as main


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """TestClient con MODELS_DIR / CONFIGS_DIR apuntando a tmp_path."""
    models = tmp_path / "models"
    configs = tmp_path / "configs"
    models.mkdir()
    configs.mkdir()
    monkeypatch.setattr(main, "MODELS_DIR", models)
    monkeypatch.setattr(main, "CONFIGS_DIR", configs)
    return TestClient(main.app), models, configs


# ── GET /models ─────────────────────────────────────────────────────────────

def test_list_models_empty(ctx):
    client, _, _ = ctx
    r = client.get("/models")
    assert r.status_code == 200
    assert r.json() == {"models": []}


def test_list_models_marks_has_config(ctx):
    client, models, configs = ctx
    (models / "foo.onnx").write_bytes(b"x")
    (configs / "foo.json").write_text("{}")
    (models / "bar.pt").write_bytes(b"x")  # sin config
    (models / "ignore.txt").write_bytes(b"x")  # extension no soportada -> excluido

    r = client.get("/models")
    assert r.status_code == 200
    by_name = {m["file"]: m for m in r.json()["models"]}
    assert set(by_name) == {"foo.onnx", "bar.pt"}
    assert by_name["foo.onnx"]["hasConfig"] is True
    assert by_name["foo.onnx"]["ext"] == "onnx"
    assert by_name["foo.onnx"]["baseName"] == "foo"
    assert by_name["bar.pt"]["hasConfig"] is False


# ── GET /configs/{name} ─────────────────────────────────────────────────────

def test_read_config_missing_returns_null(ctx):
    client, _, _ = ctx
    r = client.get("/configs/nope")
    assert r.status_code == 200
    assert r.json() == {"config": None}


def test_read_config_existing(ctx):
    client, _, configs = ctx
    (configs / "foo.json").write_text(json.dumps({"hello": "world"}))
    r = client.get("/configs/foo")
    assert r.status_code == 200
    assert r.json() == {"config": {"hello": "world"}}


def test_read_config_unsafe_name(ctx):
    client, _, _ = ctx
    # El punto no esta en [A-Za-z0-9_-] -> nombre inseguro -> 422.
    r = client.get("/configs/bad.name")
    assert r.status_code == 422


def test_read_config_corrupt_is_500(ctx):
    client, _, configs = ctx
    (configs / "broken.json").write_text("{ not valid json")
    r = client.get("/configs/broken")
    assert r.status_code == 500


# ── POST /models/upload ─────────────────────────────────────────────────────

def test_upload_valid_writes_file(ctx):
    client, models, _ = ctx
    r = client.post("/models/upload", files={"file": ("modelo.onnx", b"PESOS", "application/octet-stream")})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "file": "modelo.onnx"}
    assert (models / "modelo.onnx").read_bytes() == b"PESOS"


def test_upload_bad_extension_rejected(ctx):
    client, models, _ = ctx
    r = client.post("/models/upload", files={"file": ("modelo.txt", b"x", "text/plain")})
    assert r.status_code == 422
    assert not (models / "modelo.txt").exists()


def test_upload_unsafe_name_rejected(ctx):
    client, models, _ = ctx
    # stem "modelo.raro" contiene un punto -> nombre inseguro -> 422 (no se escribe).
    r = client.post("/models/upload", files={"file": ("modelo.raro.onnx", b"x", "application/octet-stream")})
    assert r.status_code == 422


def test_upload_strips_path_components(ctx):
    client, models, _ = ctx
    # Un filename con componentes de ruta debe quedar reducido a su basename seguro.
    r = client.post("/models/upload", files={"file": ("../../evil.onnx", b"x", "application/octet-stream")})
    assert r.status_code == 200
    assert (models / "evil.onnx").exists()


# ── GET /config/template/{type} + POST /configs/{name} (slice 4, sin test previo) ──

def test_config_template_detection_has_anchor_defaults(ctx):
    client, _, _ = ctx
    r = client.get("/config/template/detection")
    assert r.status_code == 200
    body = r.json()
    assert body["config"]["model_type"] == "detection"
    assert body["anchor_defaults"] is not None


def test_config_template_classification_no_anchors(ctx):
    client, _, _ = ctx
    r = client.get("/config/template/classification")
    assert r.status_code == 200
    assert r.json()["anchor_defaults"] is None


def test_config_template_unknown_type_404(ctx):
    client, _, _ = ctx
    r = client.get("/config/template/banana")
    assert r.status_code == 404


def test_write_config_roundtrip(ctx):
    client, _, configs = ctx
    template = client.get("/config/template/detection").json()["config"]
    r = client.post("/configs/mimodelo", json=template)
    assert r.status_code == 200
    assert (configs / "mimodelo.json").exists()


def test_write_config_invalid_body_422(ctx):
    client, _, _ = ctx
    r = client.post("/configs/mimodelo", json={"model_type": "detection"})  # falta input/output/runtime
    assert r.status_code == 422


def test_write_config_unsafe_name_422(ctx):
    client, _, _ = ctx
    template = client.get("/config/template/detection").json()["config"]
    r = client.post("/configs/bad.name", json=template)
    assert r.status_code == 422
