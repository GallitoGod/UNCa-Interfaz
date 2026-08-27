# render/draw_config.py — los ajustes de dibujo del backend.
#
# Desde el 2026-08-26 el que dibuja es el backend (paso 3 del plan del 2026-08-21),
# asi que los colores dejaron de ser un asunto del cliente. Esto es, en los hechos,
# la resurreccion del viejo /config/colors que se habia eliminado cuando el dibujo
# se habia mudado al cliente (Reforma 3).
#
# El cliente sigue siendo DUENO del estado (lo persiste en localStorage) y hace push
# aca al cambiar un ajuste y al cargar un modelo (por si el backend se reinicio).
# Aca solo vive la copia vigente que consume el hot path.
#
# Por que un singleton y no un campo del config del modelo: los colores son del
# USUARIO, no del modelo. Cambiar de modelo no debe resetearlos.

from dataclasses import dataclass, replace
from threading import Lock
from typing import Tuple

# Estilos de caja ofrecidos. Se dejan CUATRO a proposito de una familia de doce:
# un selector de doce opciones es exactamente el ruido que el wizard de modelos ya
# tuvo que podar en junio. Cada uno resuelve un caso real:
#   box    -> el rectangulo de siempre.
#   round  -> el mismo rectangulo con esquinas redondeadas (mas legible sobre texturas).
#   corner -> solo las esquinas: deja ver la imagen de abajo cuando hay muchas cajas
#             superpuestas, que es literalmente para lo que existe la app.
#   dot    -> un punto en el centro: la unica forma legible de mirar un modelo que
#             dispara cientos de detecciones chicas, donde el rectangulo tapa al objeto.
BOX_STYLES = ("box", "round", "corner", "dot")


@dataclass(frozen=True)
class DrawConfig:
    """
    Ajustes de dibujo vigentes. Inmutable: cada cambio produce una instancia nueva
    con 'version' incrementada, y esa version es parte de la clave del cache de
    annotators (ver render/annotators.py). Asi el hot path nunca construye nada
    mientras los ajustes no cambien, y cuando cambian se entera solo.

    Los colores viajan como '#RRGGBB' (lo que produce un <input type=color>) y se
    validan en el ENDPOINT, no aca: el hot path no valida.
    """
    bbox_color: str = "#00BFFF"     # default historico del cliente
    label_color: str = "#001018"    # oscuro legible sobre el fondo cian de la etiqueta
    mask_alpha: float = 0.5         # segmentacion (todavia sin pipeline)
    box_style: str = "box"          # uno de BOX_STYLES

    # Sombreado: rellena el interior de la caja con el color de acento translucido
    # (sv.ColorAnnotator). NO es la mascara de segmentacion —eso es mask_alpha y
    # necesita geometria real por pixel—, es la caja pintada por dentro: sirve para
    # que la deteccion se lea de un vistazo sin perder el detalle de abajo.
    #
    # Nace APAGADO: prende bien con "corner" (que a proposito no cierra el contorno)
    # pero sobre un frame con muchas cajas superpuestas los rellenos se suman y la
    # imagen desaparece bajo el color. Que lo prenda quien lo quiera mirar.
    shading: bool = False
    # 0.25 y no el 0.5 de supervision: a 0.5 el relleno gana sobre la foto y deja de
    # verse el objeto, que es justo lo que el usuario esta mirando.
    shading_alpha: float = 0.25

    # Etiquetas que se corren solas para no taparse entre si. Prendido por defecto:
    # con pocas cajas no se nota y con muchas es la diferencia entre leer los nombres
    # o ver una banda de carteles pisados. Cuesta ~0,37 ms con 6 cajas y crece con la
    # cantidad, por eso el cliente puede apagarlo.
    smart_labels: bool = True

    # Grosor y escala del texto derivados de la RESOLUCION del frame en vez de fijos.
    # Con valores fijos, un frame 1080p se dibuja con cajas de hilo y un 320x240 con
    # el texto al doble de tamano tapando la imagen. Con auto_scale=False se usan los
    # valores manuales de abajo.
    auto_scale: bool = True
    thickness: int = 2              # solo si auto_scale=False
    text_scale: float = 0.5         # solo si auto_scale=False

    # ── Tier B: lo que necesita memoria entre frames ──────────────────────────
    # OJO: estos toggles son del USUARIO y viven aca (persisten, sobreviven al
    # cambio de modelo), pero la MEMORIA que habilitan es por conexion y vive en
    # render/session.py. Separar las dos cosas es lo que evita inventar un tercer
    # dueno de ajustes: lo que muere con la sesion no es el toggle, es el recuerdo.
    #
    # Nace apagado: rastrear no aporta nada mirando una foto, y sobre video es una
    # herramienta de inspeccion que el usuario prende cuando quiere responder "¿mi
    # modelo pierde el objeto entre frames?".
    tracking: bool = False

    # Promedia la posicion de cada objeto en los ultimos n frames, POR IDENTIDAD.
    # Nace apagado y no por costo (es gratis al lado del tracking) sino por honestidad:
    # suavizar es MAQUILLAR al modelo, y esto es un banco de pruebas. Ademas no sale
    # gratis en calidad: promediar arrastra la caja unos px por detras del objeto en
    # movimiento sostenido (medido: ~13 px con n=5 sobre algo que avanza 5 px/frame).
    # Por eso la UI tiene que decir "suavizado n=5" y no "mejorar deteccion".
    smoothing: bool = False
    smoothing_length: int = 5

    # Estela del recorrido de cada objeto rastreado sobre los ultimos n frames.
    # Como herramienta de diagnostico vale mas de lo que parece: una traza que salta
    # de un objeto a otro muestra a simple vista que el tracker confunde identidades,
    # y una entrecortada muestra que el detector pierde el objeto en algunos frames.
    # Es la forma mas rapida de VER la estabilidad temporal de un modelo, que en
    # numeros es aburrida y en pantalla es obvia. Tambien REQUIERE tracking: sin
    # tracker_id el annotator de supervision no avisa, levanta ValueError.
    traces: bool = False
    traces_length: int = 30

    # Calidad del re-encode JPEG del frame compuesto. El frame ya llego comprimido a
    # 0.8 desde el cliente, asi que esto es una SEGUNDA compresion: es perdida de
    # calidad, no de latencia. Configurable para poder subirla si se ve degradacion.
    jpeg_quality: int = 85
    version: int = 0


_lock = Lock()

# Contador MONOTONO de versiones. No se deriva de _current.version a proposito: la
# version es la clave del cache de annotators, asi que tiene que identificar un estado
# de configuracion de forma unica durante toda la vida del proceso. Si reset() volviera
# a 0, el cache serviria annotators viejos construidos con OTRA config que tenia ese
# mismo numero (lo destapo un test que resetea entre casos).
_version_seq = 0
_current = DrawConfig()


def get_draw_config() -> DrawConfig:
    """Snapshot atomico de los ajustes vigentes. Barato: devuelve la instancia inmutable."""
    with _lock:
        return _current


def update_draw_config(**patch) -> DrawConfig:
    """
    Aplica un patch parcial y devuelve la config nueva (con version+1).
    Ignora las claves en None para que el endpoint pueda mandar solo lo que cambio.
    OJO: False NO es None, asi que apagar un booleano si se aplica.
    """
    global _current, _version_seq
    clean = {k: v for k, v in patch.items() if v is not None and k != "version"}

    # Coherencia de dependencias, aplicada ACA porque esta es la unica puerta de
    # escritura del singleton: asi no existe forma de dejar el estado en una
    # combinacion imposible, la valide quien la valide.
    #
    # El suavizado y las trazas trabajan POR tracker_id, asi que sin tracking no
    # tienen con que. Y fallan distinto, lo cual es peor que si fallaran igual: el
    # suavizado no suaviza y avisa (toggle prendido sin efecto, el sintoma que el
    # catalogo prohibe), mientras que el TraceAnnotator directamente levanta
    # ValueError y rompe el frame. En vez de dejar que el usuario arme ese estado,
    # se corrige el estado y el endpoint le devuelve el EFECTIVO, para que la UI
    # pueda mostrar lo que realmente paso.
    #
    # Si un mismo patch pide las dos cosas a la vez, apagar gana sobre prender: es
    # el pedido mas explicito ("no quiero seguimiento") y el que no deja nada
    # prendido a medias.
    if clean.get("tracking") is False:
        clean["smoothing"] = False
        clean["traces"] = False
    if clean.get("smoothing") or clean.get("traces"):
        clean["tracking"] = True

    with _lock:
        _version_seq += 1
        _current = replace(_current, version=_version_seq, **clean)
        return _current


def reset_draw_config() -> DrawConfig:
    """
    Vuelve a los defaults (lo usan los tests para no arrastrar estado entre casos).
    La version NO vuelve atras: avanza, como en cualquier otro cambio.
    """
    global _current, _version_seq
    with _lock:
        _version_seq += 1
        _current = replace(DrawConfig(), version=_version_seq)
        return _current


def hex_to_bgr(color_hex: str) -> Tuple[int, int, int]:
    """'#RRGGBB' -> (B, G, R) para OpenCV. Sin validar: eso es del endpoint."""
    h = color_hex.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
