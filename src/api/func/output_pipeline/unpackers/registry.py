# api/func/output_pipeline/unpackers/registry.py
from __future__ import annotations
from typing import Callable, Any, Dict
import numpy as np

from .raw import build_raw
from .yolo_flat import build_yolo_flat
from .boxes_scores import build_boxes_scores
from .tflite_detpost import build_tflite_detpost
from .anchor_deltas import build_anchor_deltas
from .classification import build_softmax_out, build_sigmoid_out, build_logits_raw

# El registry sirve a DOS familias de modelos. Las claves no se solapan y el
# schema ya restringe que pack_format de cada tipo (el Literal de DetectionOutput
# no acepta los de clasificacion, ni al reves), asi que una sola tabla alcanza.
#
# El CONTRATO DE FORMA de salida es distinto por familia y es responsabilidad de
# tasks/<tipo>.py conocerlo:
#   - deteccion:     ndarray 2D (N, K) -> una fila por caja candidata
#   - clasificacion: ndarray 1D (C,)   -> un puntaje por clase
UNPACKERS_REGISTRY: Dict[str, Callable] = {
    # --- deteccion ---
    "raw": build_raw,
    "yolo_flat": build_yolo_flat,
    "boxes_scores": build_boxes_scores,
    "tflite_detpost": build_tflite_detpost,
    "anchor_deltas": build_anchor_deltas,
    # --- clasificacion ---
    "softmax_out": build_softmax_out,
    "sigmoid_out": build_sigmoid_out,
    "logits_raw": build_logits_raw,
}

def unpack_out(output_cfg):
    fmt = (getattr(output_cfg, "pack_format", None) or "raw").lower()
    factory = UNPACKERS_REGISTRY.get(fmt, build_raw)
    return factory(output_cfg)
