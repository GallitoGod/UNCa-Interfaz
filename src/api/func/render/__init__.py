# render/ — composicion del frame en el backend (paso 3 del plan del 2026-08-21).
# El cliente dejo de dibujar: recibe el JPEG ya compuesto.
from .draw_config import (
    BOX_STYLES,
    DrawConfig,
    get_draw_config,
    update_draw_config,
    reset_draw_config,
    hex_to_bgr,
)
from .annotators import Annotators, annotators_for
from .session import StreamSession

__all__ = [
    "BOX_STYLES",
    "DrawConfig",
    "get_draw_config",
    "update_draw_config",
    "reset_draw_config",
    "hex_to_bgr",
    "Annotators",
    "annotators_for",
    "StreamSession",
]
