# test_stream_session.py — la memoria por conexion del stream (Tier B, paso 1).
#
# Cubre el ANDAMIAJE: el ciclo de vida de StreamSession y el contador de generacion
# del controller que lo dispara. Todavia no hay tracker adentro — eso es a proposito:
# el reseteo es la parte del Tier B que hay que poder verificar SIN que se mezcle con
# el comportamiento de un tracker, porque es lo que decide si un cambio de modelo
# arrastra identidades viejas.

import logging
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.mainAPI as main
from api.func.logger import PerfMeter
from api.func.model_controller import ModelController
from api.func.render import (
    DrawConfig,
    StreamSession,
    get_draw_config,
    reset_draw_config,
    update_draw_config,
)
from api.func.tasks.detection import _labels_for, render_detection
from api.func.tasks.domain import detections_from_array, empty_detections


@pytest.fixture
def dets():
    return detections_from_array(
        np.array([[10, 10, 50, 50, 0.9, 17], [60, 20, 90, 70, 0.8, 17]], dtype=np.float32))


# ── Ciclo de vida de la sesion ──────────────────────────────────────────────

def test_sesion_nace_sin_generacion_y_stateful():
    """Una sesion recien creada no esta atada a ningun pipeline todavia."""
    s = StreamSession()
    assert s.generation is None
    assert s.stateful is True


def test_primer_sync_no_cuenta_como_reseteo():
    """La primera sincronizacion adopta la generacion, pero no habia nada que olvidar."""
    s = StreamSession()
    assert s.sync(7) is False
    assert s.generation == 7


def test_sync_repetido_con_la_misma_generacion_no_resetea():
    """El caso normal: un frame tras otro con el mismo modelo cargado."""
    s = StreamSession()
    s.sync(7)
    assert all(s.sync(7) is False for _ in range(10))


def test_cambio_de_generacion_resetea_y_adopta_la_nueva():
    """El caso que el WebSocket no detecta solo: cambiar de modelo sin cerrar la conexion."""
    s = StreamSession()
    s.sync(7)
    assert s.sync(8) is True
    assert s.generation == 8
    # y una vez adoptada, deja de resetear
    assert s.sync(8) is False


def test_one_shot_se_declara_no_stateful():
    """El camino de imagen suelta no debe recordar nada entre fotos."""
    assert StreamSession(stateful=False).stateful is False


def test_reset_no_toca_stateful():
    """'stateful' describe a la CONEXION, no al estado: olvidar no lo cambia."""
    s = StreamSession(stateful=False)
    s.reset()
    assert s.stateful is False


def test_process_es_identidad_por_ahora(dets):
    """El andamiaje esta vacio a proposito: no debe alterar las detecciones."""
    s = StreamSession()
    salida = s.process(dets, None)
    assert len(salida) == len(dets)
    assert np.array_equal(salida.xyxy, dets.xyxy)


# ── El contador de generacion del controller ────────────────────────────────

def test_generacion_arranca_en_cero():
    assert ModelController().pipeline_generation == 0


def test_descargar_avanza_la_generacion():
    """Descargar invalida lo que una sesion viniera recordando."""
    c = ModelController()
    antes = c.pipeline_generation
    c.unload_model()
    assert c.pipeline_generation > antes


def test_la_generacion_es_monotona_y_nunca_repite():
    """
    Si volviera atras, una sesion podria creerse vigente contra un pipeline distinto
    que por casualidad tuvo ese mismo numero. Es el mismo bug que tuvo la 'version'
    de DrawConfig con el cache de annotators (corregido el 2026-08-27).
    """
    c = ModelController()
    vistas = [c.pipeline_generation]
    for _ in range(5):
        c.unload_model()
        vistas.append(c.pipeline_generation)
    assert vistas == sorted(vistas)
    assert len(set(vistas)) == len(vistas)


def test_carga_fallida_tambien_invalida_la_generacion(tmp_path):
    """
    Una carga que falla deja el controller descargado (carga atomica), asi que la
    sesion tampoco puede seguir usando lo que recordaba del modelo anterior.
    """
    c = ModelController()
    antes = c.pipeline_generation
    # setup_model_logger() corre ANTES de que la carga falle, asi que este intento
    # deja un .log con el nombre del modelo inexistente: se limpia al salir para no
    # ensuciar logs/ con basura de tests.
    nombre = "modelo_inexistente_de_test"
    try:
        with pytest.raises(Exception):
            c.load_model(f"models/{nombre}.onnx")
        assert c.pipeline_generation > antes
    finally:
        # En Windows el RotatingFileHandler mantiene el archivo abierto: hay que
        # cerrarlo antes de poder borrarlo, o el unlink levanta PermissionError.
        log = logging.getLogger(nombre)
        for h in list(log.handlers):
            h.close()
            log.removeHandler(h)
        for rastro in Path("logs").glob(f"{nombre}.log*"):
            rastro.unlink(missing_ok=True)


def test_sesion_y_controller_juntos_el_ciclo_completo(dets):
    """
    El recorrido real del handler: sync() por frame, y cuando el modelo cambia la
    sesion se entera sola sin que nadie le avise desde afuera.
    """
    c = ModelController()
    s = StreamSession()

    s.sync(c.pipeline_generation)          # primer frame
    assert s.sync(c.pipeline_generation) is False   # frames siguientes, sin novedad

    c.unload_model()                        # el usuario cambia de modelo
    assert s.sync(c.pipeline_generation) is True    # la sesion lo detecta y olvida
    assert s.sync(c.pipeline_generation) is False   # y vuelve al regimen normal


# ── El cableado en el handler del WebSocket ─────────────────────────────────

def _espiar_sesiones(monkeypatch):
    """
    Reemplaza StreamSession en mainAPI por una subclase que registra cada instancia.
    Asi se verifica que el handler la CONSTRUYE como corresponde, no solo que el
    endpoint acepta el query param sin quejarse.
    """
    creadas = []

    class SesionEspia(StreamSession):
        def __init__(self, stateful=True):
            super().__init__(stateful=stateful)
            creadas.append(self)

    monkeypatch.setattr(main, "StreamSession", SesionEspia)
    return creadas


def test_ws_crea_sesion_stateful_por_defecto(monkeypatch):
    """Camara y video: el caso normal, la conexion recuerda entre frames."""
    creadas = _espiar_sesiones(monkeypatch)
    main.controller.unload_model()
    with TestClient(main.app).websocket_connect("/video_stream") as ws:
        ws.send_bytes(b"no soy un jpeg")
        ws.receive()
    assert len(creadas) == 1 and creadas[0].stateful is True


def test_ws_respeta_stateful_false_del_query_param(monkeypatch):
    """El camino one-shot de imagenes: una foto suelta no debe recordar nada."""
    creadas = _espiar_sesiones(monkeypatch)
    main.controller.unload_model()
    with TestClient(main.app).websocket_connect("/video_stream?stateful=false") as ws:
        ws.send_bytes(b"no soy un jpeg")
        ws.receive()
    assert len(creadas) == 1 and creadas[0].stateful is False


def test_ws_sigue_respondiendo_siempre_con_la_sesion_cableada():
    """
    La invariante que no se puede romper: UN mensaje por frame, pase lo que pase.
    Cablear la sesion no debe introducir un camino donde el cliente quede esperando.
    """
    main.controller.unload_model()
    with TestClient(main.app).websocket_connect("/video_stream?stateful=false") as ws:
        for _ in range(3):
            ws.send_bytes(b"\xff\xd8\xff\xe0 basura")
            msg = ws.receive()
            assert msg.get("text") is not None


# ── Tracking (Tier B, paso 2) ───────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _draw_limpio():
    """Los ajustes de dibujo son un singleton de proceso: resetear entre casos."""
    reset_draw_config()
    yield
    reset_draw_config()


def _dets(off=0, conf=0.9, n=2):
    """Detecciones que se desplazan 'off' px, para que el tracker pueda asociarlas."""
    filas = [[10 + off + i * 150, 10, 100 + off + i * 150, 100, conf, 17] for i in range(n)]
    return detections_from_array(np.array(filas, dtype=np.float32))


def _correr(sesion, cfg, conf=0.5, frames=4, n=2, conf_det=0.9):
    """Pasa varios frames por la sesion y devuelve el ultimo resultado."""
    salida = None
    for i in range(frames):
        salida = sesion.process(_dets(off=i * 5, conf=conf_det, n=n), cfg, conf)
    return salida


def test_tracking_apagado_no_toca_las_detecciones():
    s = StreamSession()
    salida = _correr(s, DrawConfig())
    assert salida.tracker_id is None
    assert s.tracking_activo is False


def test_tracking_prendido_asigna_identidades_estables():
    """Lo que el tracking compra: el mismo objeto conserva su numero entre frames."""
    s = StreamSession()
    cfg = DrawConfig(tracking=True)
    primera = None
    for i in range(5):
        salida = s.process(_dets(off=i * 5), cfg, 0.5)
        if salida.tracker_id is not None and (salida.tracker_id >= 0).all():
            if primera is None:
                primera = salida.tracker_id.copy()
    assert primera is not None, "nunca se confirmo ninguna identidad"
    assert np.array_equal(salida.tracker_id, primera), "las identidades cambiaron de frame a frame"


def test_el_tracker_nunca_descarta_detecciones():
    """
    Prender el tracking no puede hacer desaparecer una caja. Las que todavia no
    tienen identidad confirmada salen igual, con tracker_id = -1.
    """
    s = StreamSession()
    cfg = DrawConfig(tracking=True)
    for i in range(4):
        entrada = _dets(off=i * 5, n=3)
        salida = s.process(entrada, cfg, 0.5)
        assert len(salida) == len(entrada) == 3


def test_umbrales_del_tracker_salen_del_umbral_del_usuario():
    """
    REGRESION del hallazgo que motivo todo esto: con los defaults del paquete
    (activation 0.7 / high_conf 0.6) una deteccion de 0.5 nunca recibe identidad, y
    los tres configs del repo usan umbrales de 0.5 / 0.3 / 0.25. El toggle quedaria
    prendido sin hacer nada. Derivandolos del umbral del usuario, funciona.
    """
    s = StreamSession()
    salida = _correr(s, DrawConfig(tracking=True), conf=0.3, conf_det=0.5, frames=5)
    assert salida.tracker_id is not None
    assert (salida.tracker_id >= 0).all(), (
        "una deteccion por encima del umbral del usuario debe poder recibir identidad")


def test_one_shot_nunca_rastrea_aunque_el_toggle_este_prendido():
    """Una foto suelta no es una secuencia: no hay nada que asociar entre frames."""
    s = StreamSession(stateful=False)
    salida = _correr(s, DrawConfig(tracking=True))
    assert salida.tracker_id is None
    assert s.tracking_activo is False


def test_apagar_el_tracking_suelta_el_tracker():
    """El objeto no debe quedar retenido cuando nadie lo usa (mismo criterio que shade)."""
    s = StreamSession()
    _correr(s, DrawConfig(tracking=True))
    assert s.tracking_activo is True
    salida = s.process(_dets(), DrawConfig(tracking=False), 0.5)
    assert s.tracking_activo is False
    assert salida.tracker_id is None


def test_cambiar_de_modelo_borra_las_identidades():
    """
    El caso que el WebSocket no detecta solo. Sin esto, los tracks de yolov7 seguirian
    vivos despues de cargar efficientdet, donde los class_id significan otra cosa.
    """
    s = StreamSession()
    cfg = DrawConfig(tracking=True)
    _correr(s, cfg, frames=6)
    assert s.tracking_activo is True
    s.sync(1)          # primera sincronizacion: adopta, no resetea
    assert s.sync(2) is True
    assert s.tracking_activo is False, "el tracker debe soltarse al cambiar de pipeline"


def test_mover_el_umbral_reconstruye_el_tracker():
    """
    Los umbrales del tracker se fijan al construirlo, asi que un cambio del slider
    obliga a rehacerlo. Reiniciar identidades es correcto: cambio el conjunto de
    detecciones que se esta rastreando.
    """
    s = StreamSession()
    cfg = DrawConfig(tracking=True)
    _correr(s, cfg, conf=0.5)
    primero = s._tracker
    s.process(_dets(), cfg, 0.25)
    assert s._tracker is not primero


def test_frame_vacio_igual_pasa_por_el_tracker():
    """Un frame sin detecciones es informacion: los tracks vivos tienen que envejecer."""
    s = StreamSession()
    cfg = DrawConfig(tracking=True)
    _correr(s, cfg)
    salida = s.process(empty_detections(), cfg, 0.5)
    assert len(salida) == 0
    assert s.tracking_activo is True


# ── Las etiquetas con identidad ─────────────────────────────────────────────

def test_etiqueta_antepone_el_id_cuando_hay_tracking():
    d = _dets(n=1)
    d.tracker_id = np.array([7])
    d.data["class_name"] = np.array(["horse"])
    assert _labels_for(d) == ["#7 horse 0.90"]


def test_etiqueta_omite_el_id_no_confirmado():
    """'#-1' no seria informacion, seria ruido que el usuario aprende a ignorar."""
    d = _dets(n=1)
    d.tracker_id = np.array([-1])
    d.data["class_name"] = np.array(["horse"])
    assert _labels_for(d) == ["horse 0.90"]


def test_etiqueta_sin_tracking_no_cambia():
    """Regresion: con el tracking apagado la etiqueta es exactamente la de antes."""
    d = _dets(n=1)
    d.data["class_name"] = np.array(["horse"])
    assert _labels_for(d) == ["horse 0.90"]


# ── El endpoint y las metricas ──────────────────────────────────────────────

def test_endpoint_draw_acepta_tracking_y_devuelve_el_estado_efectivo():
    client = TestClient(main.app)
    r = client.post("/config/draw", json={"tracking": True})
    assert r.status_code == 200
    assert r.json()["draw"]["tracking"] is True
    # y se puede apagar (False no es None: el patch parcial debe aplicarlo)
    assert client.post("/config/draw", json={"tracking": False}).json()["draw"]["tracking"] is False


def test_tracking_nace_apagado():
    assert get_draw_config().tracking is False


def test_perfmeter_tiene_bucket_propio_de_tracking():
    """
    Bucket separado por el mismo motivo que draw_ms: es un costo OPCIONAL y hay que
    poder mirarlo aislado para decidir si vale lo que cuesta.
    """
    p = PerfMeter(window=10)
    p.push(1.0, 2.0, 3.0, 6.0)
    p.push_track(1.5)
    s = p.stats()
    assert s["track_avg_ms"] == pytest.approx(1.5)
    # y suma al total de punta a punta que muestra el cliente
    assert s["avg_with_draw_ms"] == pytest.approx(s["avg_ms"] + s["draw_avg_ms"] + 1.5)


def test_metricas_sin_tracking_reportan_cero():
    p = PerfMeter(window=10)
    p.push(1.0, 2.0, 3.0, 6.0)
    assert p.stats()["track_avg_ms"] == 0.0


# ── Suavizado (Tier B, paso 3) ──────────────────────────────────────────────

def test_suavizado_apagado_no_promedia_nada():
    s = StreamSession()
    salida = _correr(s, DrawConfig(tracking=True))
    assert s.suavizado_activo is False
    assert salida is not None


def test_suavizado_arrastra_la_caja_detras_del_objeto():
    """
    Lo que el suavizado hace y lo que CUESTA, en el mismo test: promediar quita el
    temblequeo pero deja la caja por detras de un objeto en movimiento sostenido.
    Es la contracara honesta que la UI tiene que comunicar.
    """
    cfg = DrawConfig(tracking=True, smoothing=True, smoothing_length=5)
    s = StreamSession()
    crudo = StreamSession()
    for i in range(10):
        d = _dets(off=i * 20, n=1)
        suave = s.process(d, cfg, 0.5)
        sin_suavizar = crudo.process(d, DrawConfig(tracking=True), 0.5)
    assert s.suavizado_activo is True
    assert suave.xyxy[0][0] < sin_suavizar.xyxy[0][0], (
        "la caja suavizada debe quedar por detras de la cruda en movimiento sostenido")


def test_suavizado_no_fusiona_las_detecciones_sin_identidad():
    """
    REGRESION del bug que motivo partir las detecciones: sv.DetectionsSmoother agrupa
    por tracker_id y TODOS los tracks sin confirmar comparten el -1, asi que los toma
    por el mismo objeto. Sin partirlas, dos cajas separadas se funden en una sola en
    el medio de la nada (verificado: entran 2, sale 1).
    """
    cfg = DrawConfig(tracking=True, smoothing=True)
    s = StreamSession()
    # Los primeros frames son justo cuando conviven ids confirmados y sin confirmar.
    for i in range(4):
        entrada = _dets(off=i * 5, n=3)
        salida = s.process(entrada, cfg, 0.5)
        assert len(salida) == len(entrada) == 3, (
            f"frame {i}: entraron {len(entrada)} detecciones y salieron {len(salida)}")


def test_suavizado_directo_sobre_detecciones_sin_identidad():
    """El caso puro del bug: todas sin confirmar, ninguna se puede fusionar con otra."""
    s = StreamSession()
    d = _dets(n=2)
    d.tracker_id = np.array([-1, -1])
    salida = s._suavizar(d, 5)
    assert len(salida) == 2
    assert np.array_equal(salida.xyxy, d.xyxy), "sin identidad no se promedia nada"


def test_cambiar_la_ventana_rehace_el_suavizador():
    cfg = DrawConfig(tracking=True, smoothing=True, smoothing_length=5)
    s = StreamSession()
    _correr(s, cfg)
    primero = s._smoother
    s.process(_dets(), DrawConfig(tracking=True, smoothing=True, smoothing_length=10), 0.5)
    assert s._smoother is not primero


def test_apagar_el_suavizado_suelta_el_suavizador():
    s = StreamSession()
    _correr(s, DrawConfig(tracking=True, smoothing=True))
    assert s.suavizado_activo is True
    s.process(_dets(), DrawConfig(tracking=True), 0.5)
    assert s.suavizado_activo is False


def test_cambiar_de_modelo_borra_tambien_el_suavizador():
    s = StreamSession()
    _correr(s, DrawConfig(tracking=True, smoothing=True))
    s.sync(1)
    assert s.sync(2) is True
    assert s.suavizado_activo is False and s.tracking_activo is False


def test_one_shot_no_suaviza():
    s = StreamSession(stateful=False)
    _correr(s, DrawConfig(tracking=True, smoothing=True))
    assert s.suavizado_activo is False


# ── La dependencia suavizado -> tracking, en el singleton ───────────────────

def test_pedir_suavizado_prende_el_tracking_solo():
    """
    Regla 3 del catalogo: las dependencias se muestran, no se adivinan. Lo que NO
    puede pasar es que el toggle quede prendido sin hacer nada, que es exactamente
    lo que hace el smoother de supervision cuando le falta el tracker_id.
    """
    cfg = update_draw_config(smoothing=True)
    assert cfg.smoothing is True and cfg.tracking is True


def test_apagar_el_tracking_apaga_el_suavizado():
    update_draw_config(smoothing=True)
    cfg = update_draw_config(tracking=False)
    assert cfg.tracking is False and cfg.smoothing is False


def test_apagar_el_suavizado_deja_el_tracking_prendido():
    """Rastrear sin suavizar es un estado legitimo: es el modo de inspeccion honesto."""
    update_draw_config(smoothing=True)
    cfg = update_draw_config(smoothing=False)
    assert cfg.smoothing is False and cfg.tracking is True


def test_pedido_contradictorio_gana_el_apagado():
    """Si un mismo patch pide las dos cosas, apagar gana: no deja nada a medias."""
    cfg = update_draw_config(tracking=False, smoothing=True)
    assert cfg.tracking is False and cfg.smoothing is False


def test_endpoint_devuelve_el_estado_efectivo_no_el_pedido():
    """El cliente tiene que poder VER que el tracking se prendio solo."""
    client = TestClient(main.app)
    r = client.post("/config/draw", json={"smoothing": True})
    assert r.status_code == 200
    draw = r.json()["draw"]
    assert draw["smoothing"] is True
    assert draw["tracking"] is True, "el endpoint debe reportar el tracking que prendio solo"


def test_endpoint_valida_la_ventana_del_suavizado():
    client = TestClient(main.app)
    assert client.post("/config/draw", json={"smoothingLength": 1}).status_code == 422
    assert client.post("/config/draw", json={"smoothingLength": 99}).status_code == 422
    assert client.post("/config/draw", json={"smoothingLength": 8}).json()["draw"]["smoothingLength"] == 8


def test_suavizado_nace_apagado():
    cfg = get_draw_config()
    assert cfg.smoothing is False and cfg.smoothing_length == 5


# ── Trazas (Tier B, paso 4) ─────────────────────────────────────────────────

@pytest.fixture
def escena():
    """Frame BGR con contenido, para notar si un annotator dibujo."""
    img = np.full((300, 500, 3), 40, dtype=np.uint8)
    img[100:200, 150:350] = (200, 180, 160)
    return img


def _pintados(a, b):
    """Cuantos pixeles difieren entre dos frames."""
    return int((a != b).any(axis=2).sum())


def test_trazas_apagadas_no_construyen_el_annotator(escena):
    s = StreamSession()
    salida = s.anotar_trazas(escena, _dets(), DrawConfig(tracking=True), (500, 300))
    assert s.trazas_activas is False
    assert salida is escena, "sin trazas la escena debe pasar intacta"


def test_trazas_prendidas_dibujan_la_estela(escena):
    """Un objeto que se mueve tiene que dejar rastro a lo largo de varios frames."""
    s = StreamSession()
    cfg = DrawConfig(tracking=True, traces=True)
    for i in range(8):
        d = s.process(_dets(off=i * 25, n=1), cfg, 0.5)
        salida = s.anotar_trazas(escena.copy(), d, cfg, (500, 300))
    assert s.trazas_activas is True
    assert _pintados(salida, escena) > 0, "la estela no dibujo ni un pixel"


def test_la_estela_ignora_los_tracks_sin_confirmar(escena):
    """
    Todos los tracks sin confirmar comparten el -1, asi que una estela que los tome
    saltaria de un objeto a otro — que es exactamente el sintoma que el usuario
    deberia leer como "el tracker confunde identidades". Un artefacto que imita al
    bug que la herramienta sirve para detectar es peor que no tener la herramienta.
    """
    s = StreamSession()
    cfg = DrawConfig(tracking=True, traces=True)
    d = _dets(n=2)
    d.tracker_id = np.array([-1, -1])
    salida = s.anotar_trazas(escena.copy(), d, cfg, (500, 300))
    assert _pintados(salida, escena) == 0, "no debe dibujarse estela para tracks sin identidad"


def test_con_identidad_filtra_los_menos_uno():
    d = _dets(n=3)
    d.tracker_id = np.array([4, -1, 9])
    filtradas = StreamSession._con_identidad(d)
    assert len(filtradas) == 2
    assert list(filtradas.tracker_id) == [4, 9]


def test_con_identidad_sin_tracker_id_no_devuelve_nada():
    """Sin tracking no hay identidades: no se puede trazar nada."""
    assert len(StreamSession._con_identidad(_dets(n=2))) == 0


def test_one_shot_nunca_traza(escena):
    s = StreamSession(stateful=False)
    cfg = DrawConfig(tracking=True, traces=True)
    salida = s.anotar_trazas(escena, _dets(), cfg, (500, 300))
    assert s.trazas_activas is False and salida is escena


def test_apagar_las_trazas_suelta_el_annotator(escena):
    s = StreamSession()
    cfg = DrawConfig(tracking=True, traces=True)
    for i in range(4):
        d = s.process(_dets(off=i * 25, n=1), cfg, 0.5)
        s.anotar_trazas(escena.copy(), d, cfg, (500, 300))
    assert s.trazas_activas is True
    s.anotar_trazas(escena.copy(), d, DrawConfig(tracking=True), (500, 300))
    assert s.trazas_activas is False


def test_cambiar_de_modelo_borra_los_recorridos(escena):
    s = StreamSession()
    cfg = DrawConfig(tracking=True, traces=True)
    for i in range(4):
        d = s.process(_dets(off=i * 25, n=1), cfg, 0.5)
        s.anotar_trazas(escena.copy(), d, cfg, (500, 300))
    s.sync(1)
    assert s.sync(2) is True
    assert s.trazas_activas is False


def test_render_sin_sesion_sigue_funcionando(escena):
    """La sesion es opcional: el render tiene que componer igual sin ella."""
    jpeg = render_detection(detections_from_array(
        np.array([[10, 10, 90, 90, 0.9, 17]], dtype=np.float32)), escena, DrawConfig())
    assert isinstance(jpeg, bytes) and len(jpeg) > 0


def test_render_con_cero_detecciones_no_toca_el_frame_original(escena):
    """
    El camino sin detecciones NO copia el frame (ahorro deliberado). Si algun
    annotator escribiera ahi, estaria mutando un frame que no es nuestro.
    """
    original = escena.copy()
    s = StreamSession()
    render_detection(empty_detections(), escena,
                     DrawConfig(tracking=True, traces=True), s)
    assert np.array_equal(escena, original), "el render mutó el frame recibido"


def test_render_dibuja_la_estela_cuando_hay_sesion(escena):
    """Con y sin trazas el frame compuesto tiene que ser distinto."""
    cfg_sin = DrawConfig(tracking=True)
    cfg_con = DrawConfig(tracking=True, traces=True)
    s_sin, s_con = StreamSession(), StreamSession()
    for i in range(8):
        d_sin = s_sin.process(_dets(off=i * 25, n=1), cfg_sin, 0.5)
        d_con = s_con.process(_dets(off=i * 25, n=1), cfg_con, 0.5)
        sin = render_detection(d_sin, escena, cfg_sin, s_sin)
        con = render_detection(d_con, escena, cfg_con, s_con)
    assert sin != con, "prender las trazas debe cambiar el frame compuesto"


# ── La dependencia trazas -> tracking ───────────────────────────────────────

def test_pedir_trazas_prende_el_tracking_solo():
    """Sin tracker_id el TraceAnnotator no avisa: levanta ValueError y rompe el frame."""
    cfg = update_draw_config(traces=True)
    assert cfg.traces is True and cfg.tracking is True


def test_apagar_el_tracking_apaga_las_trazas():
    update_draw_config(traces=True, smoothing=True)
    cfg = update_draw_config(tracking=False)
    assert cfg.tracking is False and cfg.traces is False and cfg.smoothing is False


def test_endpoint_acepta_trazas_y_valida_su_ventana():
    client = TestClient(main.app)
    draw = client.post("/config/draw", json={"traces": True}).json()["draw"]
    assert draw["traces"] is True and draw["tracking"] is True
    assert client.post("/config/draw", json={"tracesLength": 1}).status_code == 422
    assert client.post("/config/draw", json={"tracesLength": 999}).status_code == 422
    assert client.post("/config/draw", json={"tracesLength": 60}).json()["draw"]["tracesLength"] == 60


def test_trazas_nacen_apagadas():
    cfg = get_draw_config()
    assert cfg.traces is False and cfg.traces_length == 30
