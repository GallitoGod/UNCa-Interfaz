# api/func/output_pipeline/unpackers/registry.py
from __future__ import annotations
from typing import Callable, Any, Dict
import numpy as np

from ._shape import normalize_to_2d
from .raw import build_raw
from .yolo_flat import build_yolo_flat
from .boxes_scores import build_boxes_scores
from .tflite_detpost import build_tflite_detpost
from .anchor_deltas import build_anchor_deltas

UNPACKERS_REGISTRY: Dict[str, Callable] = {
    "raw": build_raw,
    "yolo_flat": build_yolo_flat,
    "boxes_scores": build_boxes_scores,
    "tflite_detpost": build_tflite_detpost,
    "anchor_deltas": build_anchor_deltas,
}

def unpack_out(output_cfg):
    """Construye el unpacker para el pack_format declarado y lo envuelve en el
    contrato de forma UNICO: la funcion devuelta SIEMPRE entrega un ndarray 2D
    float32 (N, K). Toda la normalizacion de shapes converge aca (reforma Fase 2,
    tarea 2): ni los unpackers individuales ni el controller la repiten."""
    fmt = (getattr(output_cfg, "pack_format", None) or "raw").lower()
    factory = UNPACKERS_REGISTRY.get(fmt, build_raw)
    base = factory(output_cfg)

    def fn(raw_output: Any, runtime=None) -> np.ndarray:
        return normalize_to_2d(base(raw_output, runtime))

    return fn
