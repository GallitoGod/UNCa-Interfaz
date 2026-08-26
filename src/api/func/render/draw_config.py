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


@dataclass(frozen=True)
class DrawConfig:
    """
    Ajustes de dibujo vigentes. Inmutable: cada cambio produce una instancia nueva
    con 'version' incrementada, y esa version es la clave del cache de annotators
    (ver render/annotators.py). Asi el hot path nunca construye nada mientras los
    ajustes no cambien, y cuando cambian se entera solo.

    Los colores viajan como '#RRGGBB' (lo que produce un <input type=color>) y se
    validan en el ENDPOINT, no aca: el hot path no valida.
    """
    bbox_color: str = "#00BFFF"     # default historico del cliente
    label_color: str = "#001018"    # oscuro legible sobre el fondo cian de la etiqueta
    mask_alpha: float = 0.5         # segmentacion (todavia sin pipeline)
    thickness: int = 2
    text_scale: float = 0.5
    # Calidad del re-encode JPEG del frame compuesto. El frame ya llego comprimido a
    # 0.8 desde el cliente, asi que esto es una SEGUNDA compresion: es perdida de
    # calidad, no de latencia. Configurable para poder subirla si se ve degradacion.
    jpeg_quality: int = 85
    version: int = 0


_lock = Lock()
_current = DrawConfig()


def get_draw_config() -> DrawConfig:
    """Snapshot atomico de los ajustes vigentes. Barato: devuelve la instancia inmutable."""
    with _lock:
        return _current


def update_draw_config(**patch) -> DrawConfig:
    """
    Aplica un patch parcial y devuelve la config nueva (con version+1).
    Ignora las claves en None para que el endpoint pueda mandar solo lo que cambio.
    """
    global _current
    clean = {k: v for k, v in patch.items() if v is not None and k != "version"}
    with _lock:
        _current = replace(_current, version=_current.version + 1, **clean)
        return _current


def reset_draw_config() -> DrawConfig:
    """Vuelve a los defaults (lo usan los tests para no arrastrar estado entre casos)."""
    global _current
    with _lock:
        _current = DrawConfig()
        return _current


def hex_to_bgr(color_hex: str) -> Tuple[int, int, int]:
    """'#RRGGBB' -> (B, G, R) para OpenCV. Sin validar: eso es del endpoint."""
    h = color_hex.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
