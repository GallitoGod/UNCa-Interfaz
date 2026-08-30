# test_render_backend.py — el render en el BACKEND (paso 3 del plan del 2026-08-21).
#
# Cubre las tres piezas nuevas: la composicion del frame (render_detection), los
# ajustes de dibujo en vivo (render/draw_config + /config/draw) y el despacho del WS
# por output_kind. Lo que NO se toca aca a proposito: clasificacion — si un test de
# CLS necesitara cambiar por este paso, seria la senal de que algo se rompio.

import cv2
import numpy as np
import pytest
import supervision as sv
from fastapi.testclient import TestClient

import api.mainAPI as main
from api.func.render import (
    BOX_STYLES,
    LABEL_MODES,
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


# ── Tier A (2026-08-27): escala adaptativa, etiquetas que se esquivan, estilos ──

def test_auto_scale_deriva_grosor_y_texto_de_la_resolucion():
    # El defecto que corrige: con valores fijos, 1080p se dibuja con cajas de hilo
    # y 320x240 con el texto al doble. Los dos extremos tienen que dar distinto.
    cfg = get_draw_config()
    chico = annotators_for(cfg, (320, 240))
    grande = annotators_for(cfg, (1920, 1080))
    assert grande.thickness > chico.thickness
    assert grande.text_scale > chico.text_scale
    assert grande.thickness == sv.calculate_optimal_line_thickness((1920, 1080))
    assert grande.text_scale == sv.calculate_optimal_text_scale((1920, 1080))


def test_auto_scale_apagado_respeta_los_valores_manuales():
    cfg = update_draw_config(auto_scale=False, thickness=7, text_scale=1.25)
    ann = annotators_for(cfg, (1920, 1080))
    assert ann.thickness == 7 and ann.text_scale == 1.25


def test_cache_de_annotators_distingue_resolucion():
    cfg = get_draw_config()
    a = annotators_for(cfg, (640, 480))
    assert annotators_for(cfg, (640, 480)) is a       # misma clave -> mismo objeto
    assert annotators_for(cfg, (1280, 720)) is not a  # otra resolucion -> otro objeto


@pytest.mark.parametrize("style,clase", [
    ("box", sv.BoxAnnotator),
    ("round", sv.RoundBoxAnnotator),
    ("corner", sv.BoxCornerAnnotator),
    ("dot", sv.DotAnnotator),
])
def test_cada_estilo_construye_su_annotator(style, clase):
    ann = annotators_for(update_draw_config(box_style=style), (860, 573))
    assert isinstance(ann.box, clase)


@pytest.mark.parametrize("style", list(BOX_STYLES))
def test_render_funciona_con_todos_los_estilos(style, frame, dets):
    update_draw_config(box_style=style)
    jpg = render_detection(dets, frame)
    back = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert back is not None and back.shape == frame.shape


def test_estilo_desconocido_cae_al_rectangulo():
    # El endpoint valida, pero un config viejo no debe romper el frame: preferimos
    # dibujar algo antes que tirar la inferencia.
    ann = annotators_for(update_draw_config(box_style="triangulo_raro"), (860, 573))
    assert isinstance(ann.box, sv.BoxAnnotator)


def test_smart_labels_llega_al_annotator():
    assert annotators_for(update_draw_config(smart_labels=True), (860, 573)).label.smart_position
    assert not annotators_for(update_draw_config(smart_labels=False), (860, 573)).label.smart_position


def test_smart_labels_prendido_por_defecto():
    # Con pocas cajas no se nota y con muchas es la diferencia entre leer los nombres
    # o ver una banda de carteles pisados: el default correcto es prendido.
    assert DrawConfig().smart_labels is True
    assert DrawConfig().auto_scale is True


def test_post_config_draw_acepta_los_ajustes_nuevos():
    client = TestClient(main.app)
    r = client.post("/config/draw", json={
        "boxStyle": "corner", "smartLabels": False, "autoScale": False,
        "thickness": 5, "textScale": 0.9,
    })
    assert r.status_code == 200
    draw = r.json()["draw"]
    assert draw["boxStyle"] == "corner"
    assert draw["smartLabels"] is False and draw["autoScale"] is False
    cfg = get_draw_config()
    assert cfg.box_style == "corner" and cfg.thickness == 5


def test_apagar_un_booleano_no_se_confunde_con_no_mandarlo():
    # update_draw_config ignora los None; False tiene que APLICARSE igual.
    update_draw_config(smart_labels=True)
    TestClient(main.app).post("/config/draw", json={"smartLabels": False})
    assert get_draw_config().smart_labels is False


@pytest.mark.parametrize("body", [
    {"boxStyle": "triangulo"},   # no esta en BOX_STYLES
    {"boxStyle": "BOX"},         # sensible a mayusculas, como el resto del schema
    {"textScale": 0},            # tiene que ser > 0
    {"textScale": 9},            # fuera de rango
])
def test_post_config_draw_rechaza_ajustes_nuevos_invalidos(body):
    assert TestClient(main.app).post("/config/draw", json=body).status_code == 422


def test_la_version_nunca_vuelve_atras():
    # La version es la clave del cache de annotators: si reset() la devolviera a 0,
    # el cache serviria objetos construidos con OTRA config que tuvo ese numero.
    v = update_draw_config(bbox_color="#010203").version
    assert reset_draw_config().version > v
    assert update_draw_config(box_style="dot").version > v + 1


# ── Sombreado de la caja (2026-08-26) ─────────────────────────────────────────

def test_sombreado_apagado_no_construye_annotator():
    # Apagado tiene que costar CERO: ni objeto construido ni entrada retenida.
    assert DrawConfig().shading is False
    assert annotators_for(get_draw_config(), (860, 573)).shade is None


def test_sombreado_prendido_construye_un_color_annotator():
    ann = annotators_for(update_draw_config(shading=True), (860, 573))
    # OJO: es ColorAnnotator (rellena el RECTANGULO), no MaskAnnotator (que necesita
    # detections.mask por pixel, cosa que un detector no produce).
    assert isinstance(ann.shade, sv.ColorAnnotator)
    assert ann.shade.opacity == DrawConfig().shading_alpha


def test_shading_alpha_llega_al_annotator():
    ann = annotators_for(update_draw_config(shading=True, shading_alpha=0.6), (860, 573))
    assert ann.shade.opacity == 0.6


def test_sombreado_pinta_adentro_de_la_caja(frame, dets):
    # La prueba real: el interior de la caja tiene que cambiar respecto de no
    # sombrear. Se mira un pixel BIEN adentro (no el borde, que ya lo pinta el
    # contorno con o sin sombreado).
    update_draw_config(shading=False)
    sin = cv2.imdecode(np.frombuffer(render_detection(dets, frame), np.uint8), cv2.IMREAD_COLOR)
    update_draw_config(shading=True)
    con = cv2.imdecode(np.frombuffer(render_detection(dets, frame), np.uint8), cv2.IMREAD_COLOR)
    # dets = [10,20,90,100]; (50,60) cae en el centro, lejos del trazo.
    assert not np.array_equal(sin[60, 50], con[60, 50])


def test_sombreado_no_toca_lo_de_afuera_de_la_caja(frame, dets):
    # El relleno es de la deteccion, no del frame: fuera de [10,20,90,100] no cambia
    # ni un pixel. Si esto falla, el annotator esta pintando la escena entera.
    update_draw_config(shading=True)
    con = cv2.imdecode(np.frombuffer(render_detection(dets, frame), np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(con[110, 180], frame[110, 180])


def test_sombreado_convive_con_todos_los_estilos(frame, dets):
    # Es una capa aparte, no una propiedad del estilo: se apila abajo de cualquiera.
    for style in BOX_STYLES:
        update_draw_config(box_style=style, shading=True)
        jpg = render_detection(dets, frame)
        back = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        assert back is not None and back.shape == frame.shape


def test_sombreado_no_muta_el_frame_original(frame, dets):
    # El .copy() se movio de lugar al encadenar annotators: si alguien lo saca, el
    # frame del handler del WS —que no es nuestro— queda pintado.
    update_draw_config(shading=True)
    antes = frame.copy()
    render_detection(dets, frame)
    assert np.array_equal(frame, antes)


def test_post_config_draw_acepta_el_sombreado():
    client = TestClient(main.app)
    r = client.post("/config/draw", json={"shading": True, "shadingAlpha": 0.4})
    assert r.status_code == 200
    draw = r.json()["draw"]
    assert draw["shading"] is True and draw["shadingAlpha"] == 0.4
    cfg = get_draw_config()
    assert cfg.shading is True and cfg.shading_alpha == 0.4


def test_apagar_el_sombreado_se_aplica():
    # Mismo caso que smartLabels: False no es None, tiene que aplicarse.
    update_draw_config(shading=True)
    TestClient(main.app).post("/config/draw", json={"shading": False})
    assert get_draw_config().shading is False


@pytest.mark.parametrize("body", [
    {"shadingAlpha": -0.1},   # fuera de rango por abajo
    {"shadingAlpha": 1.5},    # fuera de rango por arriba
    {"shading": "si"},        # no es booleano
])
def test_post_config_draw_rechaza_sombreado_invalido(body):
    assert TestClient(main.app).post("/config/draw", json=body).status_code == 422


# ── Modos de etiqueta (pendiente #27, 2026-08-28) ─────────────────────────────
#
# El problema que resuelven: con 'best' (VisDrone) sobre material aereo salen ~70
# detecciones por frame y los carteles tapan la escena entera. El smart_position no
# alcanza — no es que se pisen, es que no hay lugar.

def test_label_mode_nace_en_completa():
    # Con pocas cajas el cartel entero es la mejor lectura, y es lo que el sistema
    # venia haciendo: cambiar el default seria una regresion para el caso comun.
    assert DrawConfig().label_mode == "completa"


def test_modo_completa_trae_nombre_y_confianza(dets):
    assert _labels_for(dets, "completa") == ["17 0.87"]


def test_modo_corta_omite_la_confianza(dets):
    # Es el ~40% del ancho del cartel y es un numero que se lee para UNA caja, no
    # para setenta.
    assert _labels_for(dets, "corta") == ["17"]


def test_el_prefijo_de_tracking_sobrevive_al_modo_corto():
    # El #id es lo mas corto de la etiqueta y lo unico que NO se puede deducir
    # mirando el frame: es lo ultimo que habria que sacar.
    d = detections_from_array(
        np.array([[10.0, 20.0, 90.0, 100.0, 0.87, 17.0]], dtype=np.float32))
    d.tracker_id = np.array([3])
    assert _labels_for(d, "completa") == ["#3 17 0.87"]
    assert _labels_for(d, "corta") == ["#3 17"]


def test_modo_ninguna_no_construye_el_annotator_de_etiquetas():
    # Mismo criterio que 'shade': el caso apagado no retiene un objeto que nadie usa,
    # y el hot path pregunta por None en vez de releer un flag de la config.
    assert annotators_for(update_draw_config(label_mode="ninguna"), (860, 573)).label is None
    assert annotators_for(update_draw_config(label_mode="completa"), (860, 573)).label is not None


def test_modo_ninguna_produce_un_frame_distinto(frame, dets):
    # La verificacion que importa: apagar las etiquetas tiene que cambiar PIXELES.
    # Un toggle prendido sin efecto visible es exactamente lo que este proyecto
    # viene peleando.
    update_draw_config(label_mode="completa")
    con = render_detection(dets, frame)
    update_draw_config(label_mode="ninguna")
    sin = render_detection(dets, frame)
    assert con != sin


def test_apagar_las_etiquetas_no_cambia_ninguna_deteccion(frame, dets):
    # Es una capa de dibujo, no un filtro: el resultado del modelo no se toca.
    antes = dets.xyxy.copy(), dets.confidence.copy(), dets.class_id.copy()
    update_draw_config(label_mode="ninguna")
    render_detection(dets, frame)
    assert np.array_equal(dets.xyxy, antes[0])
    assert np.array_equal(dets.confidence, antes[1])
    assert np.array_equal(dets.class_id, antes[2])


def test_los_tres_modos_componen_un_frame_valido(frame, dets):
    for modo in LABEL_MODES:
        update_draw_config(label_mode=modo)
        jpg = render_detection(dets, frame)
        back = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert back is not None and back.shape == frame.shape, modo


def test_post_config_draw_acepta_label_mode():
    client = TestClient(main.app)
    r = client.post("/config/draw", json={"labelMode": "corta"})
    assert r.status_code == 200
    assert r.json()["draw"]["labelMode"] == "corta"
    assert get_draw_config().label_mode == "corta"


def test_post_config_draw_rechaza_un_modo_inventado():
    r = TestClient(main.app).post("/config/draw", json={"labelMode": "solo_el_id"})
    assert r.status_code == 422


def test_el_endpoint_y_LABEL_MODES_no_pueden_desincronizarse():
    # La misma guarda que ya protege a boxStyle: si alguien agrega un modo en
    # draw_config y se olvida del endpoint, revienta al importar, no en produccion.
    for modo in LABEL_MODES:
        assert TestClient(main.app).post(
            "/config/draw", json={"labelMode": modo}).status_code == 200
