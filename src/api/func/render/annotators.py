# render/annotators.py — los annotators de supervision, cacheados.
#
# Los annotators son objetos con estado de CONFIGURACION (color, grosor, escala de
# texto, estilo de caja), no funciones puras: construirlos por frame seria tirar
# trabajo. Pero los ajustes cambian en vivo desde el cliente, asi que tampoco se
# pueden construir una sola vez para siempre.
#
# Solucion: se cachean contra (version de la DrawConfig, resolucion del frame).
# La resolucion entra en la clave porque el grosor y la escala del texto se derivan
# de ella (auto_scale); cambia cuando cambia la fuente, no por frame.

from dataclasses import dataclass
from threading import Lock
from typing import Optional, Tuple

import supervision as sv

from .draw_config import DrawConfig

# Cuantas combinaciones (version, resolucion) se retienen. Con tres fuentes vivas
# (camara, video, imagen) alcanza de sobra; el cap solo existe para que el cache no
# crezca sin limite si alguien alterna resoluciones raras.
_MAX_ENTRADAS = 4


@dataclass(frozen=True)
class Annotators:
    """Los annotators armados para una version de DrawConfig y una resolucion."""
    box: object            # el estilo elegido: Box / RoundBox / BoxCorner / Dot
    label: sv.LabelAnnotator
    mask: sv.MaskAnnotator
    thickness: int         # el efectivo (auto o manual), para logs y tests
    text_scale: float


_lock = Lock()
_cache: "dict[Tuple[int, Optional[Tuple[int, int]]], Annotators]" = {}


def _escala(cfg: DrawConfig, resolution_wh: Optional[Tuple[int, int]]):
    """
    Grosor y escala de texto efectivos.

    Con auto_scale, supervision los deriva de la resolucion: es la regla que Roboflow
    destilo de mirar miles de frames anotados, y evita el defecto que teniamos —
    valores fijos elegidos a ojo con UNA imagen de 860 px, que a 1080p dibujan cajas
    de hilo y a 320x240 tapan la imagen con el texto.
    """
    if cfg.auto_scale and resolution_wh is not None:
        return (sv.calculate_optimal_line_thickness(resolution_wh),
                sv.calculate_optimal_text_scale(resolution_wh))
    return cfg.thickness, cfg.text_scale


def _box_annotator(style: str, color: sv.Color, thickness: int):
    """El annotator del estilo elegido, con el grosor ya resuelto."""
    if style == "round":
        # roundness 0.25 y no el 0.6 de supervision: con cajas grandes el default
        # curva tanto que las esquinas se leen como arcos, no como una caja.
        return sv.RoundBoxAnnotator(
            color=color, thickness=thickness, roundness=0.25,
            color_lookup=sv.ColorLookup.INDEX)
    if style == "corner":
        # corner_length atado al grosor: si no, en 4K quedan esquinitas invisibles.
        return sv.BoxCornerAnnotator(
            color=color, thickness=thickness, corner_length=max(10, thickness * 6),
            color_lookup=sv.ColorLookup.INDEX)
    if style == "dot":
        # El radio va atado al grosor (que ya escala con la resolucion): con
        # thickness*2 el punto queda casi invisible en frames grandes.
        return sv.DotAnnotator(
            color=color, radius=max(4, thickness * 3), color_lookup=sv.ColorLookup.INDEX)
    # "box" y cualquier valor desconocido caen al rectangulo: el endpoint ya valida,
    # y ante un config viejo preferimos dibujar algo antes que romper el frame.
    return sv.BoxAnnotator(
        color=color, thickness=thickness, color_lookup=sv.ColorLookup.INDEX)


def _build(cfg: DrawConfig, resolution_wh: Optional[Tuple[int, int]]) -> Annotators:
    # color_lookup=INDEX y un Color unico: el usuario eligio UN color de caja, no una
    # paleta por clase. Con el default (ColorLookup.CLASS + ColorPalette) supervision
    # ignoraria el color elegido y pintaria cada clase de un color distinto.
    box_color = sv.Color.from_hex(cfg.bbox_color)
    thickness, text_scale = _escala(cfg, resolution_wh)

    return Annotators(
        box=_box_annotator(cfg.box_style, box_color, thickness),
        label=sv.LabelAnnotator(
            color=box_color,
            text_color=sv.Color.from_hex(cfg.label_color),
            text_scale=text_scale,
            smart_position=cfg.smart_labels,
            color_lookup=sv.ColorLookup.INDEX,
        ),
        mask=sv.MaskAnnotator(
            color=box_color,
            opacity=cfg.mask_alpha,
            color_lookup=sv.ColorLookup.INDEX,
        ),
        thickness=thickness,
        text_scale=text_scale,
    )


def annotators_for(cfg: DrawConfig, resolution_wh: Optional[Tuple[int, int]] = None) -> Annotators:
    """
    Devuelve los annotators de esta config y esta resolucion, reconstruyendolos solo
    si cambio alguna de las dos. resolution_wh=None desactiva el auto_scale (se usan
    los valores manuales): util para tests y para llamadas sin frame a la vista.
    """
    clave = (cfg.version, resolution_wh)
    cached = _cache.get(clave)
    if cached is not None:
        return cached
    with _lock:
        # Doble chequeo: otro hilo pudo construirlo mientras esperabamos el lock.
        cached = _cache.get(clave)
        if cached is not None:
            return cached
        if len(_cache) >= _MAX_ENTRADAS:
            _cache.clear()          # cache chico: vaciarlo es mas simple que un LRU
        annotators = _build(cfg, resolution_wh)
        _cache[clave] = annotators
        return annotators
