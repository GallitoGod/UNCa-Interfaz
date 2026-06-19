# api/func/output_pipeline/unpackers/raw.py
from __future__ import annotations
from typing import Any
import numpy as np


def build_raw(output_cfg):
    """Unpacker 'raw': el modelo ya emite filas listas para el adapter; no hay
    nada que decodificar. La normalizacion de forma (list -> 2D float32) la hace
    una sola capa: el wrapper de registry.unpack_out via normalize_to_2d. Por eso
    'raw' es un passthrough y NO duplica esa logica (reforma Fase 2, tarea 2)."""
    def fn(raw_output: Any, sh=None):
        return raw_output

    return fn
