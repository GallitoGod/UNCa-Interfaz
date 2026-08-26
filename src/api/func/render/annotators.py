# render/annotators.py — los annotators de supervision, cacheados por version.
#
# Los annotators son objetos con estado de CONFIGURACION (color, grosor, escala de
# texto), no funciones puras: construirlos por frame seria tirar trabajo. Pero los
# ajustes cambian en vivo desde el cliente, asi que tampoco se pueden construir una
# sola vez para siempre.
#
# Solucion: se cachean contra la 'version' de la DrawConfig. Mientras el usuario no
# toque un color, el hot path devuelve los mismos objetos; apenas toca uno, la
# version cambia y se reconstruyen UNA vez.

from dataclasses import dataclass
from threading import Lock

import supervision as sv

from .draw_config import DrawConfig


@dataclass(frozen=True)
class Annotators:
    """Los annotators armados para una version concreta de la DrawConfig."""
    box: sv.BoxAnnotator
    label: sv.LabelAnnotator
    mask: sv.MaskAnnotator
    version: int


_lock = Lock()
_cache: Annotators | None = None


def _build(cfg: DrawConfig) -> Annotators:
    # color_lookup=INDEX y un Color unico: el usuario eligio UN color de caja, no una
    # paleta por clase. Con el default (ColorLookup.CLASS + ColorPalette) supervision
    # ignoraria el color elegido y pintaria cada clase de un color distinto.
    box_color = sv.Color.from_hex(cfg.bbox_color)
    return Annotators(
        box=sv.BoxAnnotator(
            color=box_color,
            thickness=cfg.thickness,
            color_lookup=sv.ColorLookup.INDEX,
        ),
        label=sv.LabelAnnotator(
            color=box_color,
            text_color=sv.Color.from_hex(cfg.label_color),
            text_scale=cfg.text_scale,
            color_lookup=sv.ColorLookup.INDEX,
        ),
        mask=sv.MaskAnnotator(
            color=box_color,
            opacity=cfg.mask_alpha,
            color_lookup=sv.ColorLookup.INDEX,
        ),
        version=cfg.version,
    )


def annotators_for(cfg: DrawConfig) -> Annotators:
    """Devuelve los annotators de esta config, reconstruyendolos solo si cambio la version."""
    global _cache
    cached = _cache
    if cached is not None and cached.version == cfg.version:
        return cached
    with _lock:
        # Doble chequeo: otro hilo pudo reconstruirlos mientras esperabamos el lock.
        if _cache is not None and _cache.version == cfg.version:
            return _cache
        _cache = _build(cfg)
        return _cache
