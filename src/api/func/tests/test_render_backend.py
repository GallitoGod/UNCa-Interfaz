# test_render_backend.py — el render en el BACKEND (paso 3 del plan del 2026-08-21).
#
# Cubre las tres piezas nuevas: la composicion del frame (render_detection), los
# ajustes de dibujo en vivo (render/draw_config + /config/draw) y el despacho del WS
# por output_kind. Lo que NO se toca aca a proposito: clasificacion — si un test de
# CLS necesitara cambiar por este paso, seria la senal de que algo se rompio.

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.mainAPI as main
from api.func.render import (
    DrawConfig,
    annotators_for,
    get_draw_config,
    hex_to_bgr,
    reset_draw_config,
    update_draw_config,
)
from api.func.tasks.detection import _class_name, _labels_for, render_detection
from api.func.tasks.domain import detections_from_array, empty_detections
from api.func.tasks.registry import get_strategy


@pytest.fixture(autouse=True)
def _draw_limpio():
    """Los ajustes de dibujo son un singleton de proceso: resetear entre casos."""
    reset_draw_config()
    yield
    reset_draw_config()


@pytest.fixture
def frame():
    # Frame BGR con contenido (no negro puro): asi se nota si el annotator dibujo.
    img = np.full((120, 200, 3), 40, dtype=np.uint8)
    img[40:80, 60:140] = (200, 180, 160)
    return img


@pytest.fixture
def dets():
    return detections_from_array(
        np.array([[10.0, 20.0, 90.0, 100.0, 0.87, 17.0]], dtype=np.float32))


# ── Composicion del frame ─────────────────────────────────────────────────────

def test_render_devuelve_jpeg_decodificable_del_mismo_tamano(frame, dets):
    jpg = render_detection(dets, frame)
    assert isinstance(jpg, bytes) and jpg[:2] == b"\xff\xd8"  # SOI de JPEG
    back = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert back.shape == frame.shape


def test_render_no_muta_el_frame_original(frame, dets):
    # Los annotators de supervision escriben IN-PLACE: sin el .copy() del render
    # estariamos pisando el frame del handler del WS.
    antes = frame.copy()
    render_detection(dets, frame)
    assert np.array_equal(frame, antes)


def test_render_sin_cajas_igual_devuelve_el_frame(frame):
    # Un frame sin detecciones no es un error: se re-encodea tal cual. El WS tiene
    # que responder SIEMPRE, tambien cuando no hay nada que dibujar.
    jpg = render_detection(empty_detections(), frame)
    back = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert back.shape == frame.shape


def test_render_dibuja_algo(frame, dets):
    # Comparacion contra el frame crudo re-encodeado con la MISMA calidad: si el
    # annotator no hubiera dibujado, los bytes serian identicos.
    cfg = get_draw_config()
    _, crudo = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality])
    assert render_detection(dets, frame) != crudo.tobytes()


# ── Etiquetas ─────────────────────────────────────────────────────────────────

def test_labels_caen_al_id_numerico_sin_label_map(dets):
    assert _labels_for(dets) == ["17 0.87"]


def test_labels_usan_el_label_map_cuando_esta(dets):
    # El pipeline adjunta los nombres en data['class_name'], que es donde
    # supervision los espera; render_detection no conoce el config del modelo.
    dets.data["class_name"] = np.array(["horse"])
    assert _labels_for(dets) == ["horse 0.87"]


def test_class_name_fuera_de_rango_cae_al_id():
    # Un label_map mas corto que el numero de clases del modelo no debe reventar
    # el frame: se dibuja el id, que es informacion honesta.
    assert _class_name(["person", "bicycle"], 1) == "bicycle"
    assert _class_name(["person", "bicycle"], 40) == "40"


# ── Ajustes de dibujo ─────────────────────────────────────────────────────────

def test_hex_a_bgr():
    assert hex_to_bgr("#00BFFF") == (255, 191, 0)


def test_update_bumpea_version_y_conserva_lo_no_tocado():
    v0 = get_draw_config()
    v1 = update_draw_config(bbox_color="#FF0000")
    assert v1.version == v0.version + 1
    assert v1.bbox_color == "#FF0000"
    assert v1.label_color == v0.label_color   # lo que no se manda no se pisa


def test_annotators_se_cachean_por_version():
    cfg = get_draw_config()
    a1 = annotators_for(cfg)
    assert annotators_for(cfg) is a1           # misma version -> mismos objetos
    a2 = annotators_for(update_draw_config(thickness=6))
    assert a2 is not a1                        # version nueva -> reconstruidos
    assert a2.box.thickness == 6


def test_draw_config_es_inmutable():
    with pytest.raises(Exception):
        get_draw_config().bbox_color = "#FFFFFF"


# ── Endpoint /config/draw ─────────────────────────────────────────────────────

def test_post_config_draw_ok():
    client = TestClient(main.app)
    r = client.post("/config/draw", json={"bboxColor": "#123456", "thickness": 4})
    assert r.status_code == 200
    assert r.json()["draw"]["bboxColor"] == "#123456"
    assert get_draw_config().thickness == 4


def test_post_config_draw_no_requiere_modelo_cargado():
    # Los colores son del USUARIO, no del modelo: 409 aca seria un error de diseno.
    client = TestClient(main.app)
    assert client.post("/config/draw", json={"maskAlpha": 0.3}).status_code == 200


@pytest.mark.parametrize("body", [
    {"bboxColor": "rojo"},          # no es #RRGGBB
    {"bboxColor": "#GGGGGG"},       # no es hex
    {"maskAlpha": 1.5},             # fuera de [0,1]
    {"jpegQuality": 0},             # fuera de [1,100]
    {"thickness": 0},
])
def test_post_config_draw_rechaza_valores_invalidos(body):
    client = TestClient(main.app)
    assert client.post("/config/draw", json=body).status_code == 422


# ── Despacho del WS por output_kind ───────────────────────────────────────────

def test_estrategias_declaran_su_output_kind():
    det = get_strategy("detection")
    assert det.output_kind == "frame" and callable(det.render)
    # Clasificacion sigue siendo JSON: su resultado es texto, no geometria.
    cls = get_strategy("classification")
    assert cls.output_kind == "json" and cls.render is None


def test_ws_responde_json_sin_modelo():
    # La invariante "el WS SIEMPRE responde" no cambia con el paso 3, y los errores
    # viajan como TEXTO aunque la tarea activa dibuje.
    main.controller.unload_model()
    client = TestClient(main.app)
    with client.websocket_connect("/video_stream") as ws:
        ws.send_bytes(b"\xff\xd8\xff\xe0 no soy un jpeg valido")
        msg = ws.receive()
        assert msg.get("text") is not None, "el error debe viajar como texto, no binario"


def test_output_kind_sin_modelo_es_json():
    main.controller.unload_model()
    assert main.controller.output_kind == "json"
