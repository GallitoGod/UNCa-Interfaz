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

# ---------------------------------------------------------------------------
# CABLES Fase 2 tarea 3: pack_format de clasificacion/segmentacion RECONOCIDOS
# por el sistema pero todavia SIN logica. unpack_out devuelve un stub que levanta
# NotImplementedError. Cuando se implemente el pipeline real, mover el formato a
# UNPACKERS_REGISTRY con su builder de verdad y quitarlo de aca.
# ---------------------------------------------------------------------------
_PENDING_FORMATS: Dict[str, str] = {
    # clasificacion (ver ClassificationOutput.pack_format)
    "softmax_out": "clasificacion: el tensor ya trae probabilidades (softmax)",
    "sigmoid_out": "clasificacion multi-label: probabilidades sigmoid por clase",
    "logits_raw":  "clasificacion: logits crudos (falta aplicar softmax/sigmoid)",
    # segmentacion (ver SemanticSegmentationOutput.pack_format)
    "argmax_map":  "segmentacion: mapa de clase ganadora por pixel (HxW)",
    "softmax_map": "segmentacion: probabilidades por canal (CxHxW o HxWxC)",
    "binary_mask": "segmentacion binaria: mascara unica",
}


def _build_pending(fmt: str, descripcion: str):
    """Stub de unpacker: el formato esta reconocido pero su logica no existe aun."""
    def fn(raw_output: Any, runtime=None):
        raise NotImplementedError(
            f"unpacker '{fmt}' ({descripcion}) todavia no esta implementado. "
            "El sistema lo reconoce pero falta conectar su logica de desempaque "
            "(Fase 2 tarea 3).")
    return fn


def unpack_out(output_cfg):
    """Construye el unpacker para el pack_format declarado y lo envuelve en el
    contrato de forma UNICO: la funcion devuelta SIEMPRE entrega un ndarray 2D
    float32 (N, K). Toda la normalizacion de shapes converge aca (reforma Fase 2,
    tarea 2): ni los unpackers individuales ni el controller la repiten.

    Para formatos de CLS/SEG todavia sin implementar (_PENDING_FORMATS) devuelve un
    stub que levanta NotImplementedError: el cable esta puesto, falta la logica."""
    fmt = (getattr(output_cfg, "pack_format", None) or "raw").lower()

    if fmt in _PENDING_FORMATS:
        return _build_pending(fmt, _PENDING_FORMATS[fmt])

    factory = UNPACKERS_REGISTRY.get(fmt, build_raw)
    base = factory(output_cfg)

    def fn(raw_output: Any, runtime=None) -> np.ndarray:
        return normalize_to_2d(base(raw_output, runtime))

    return fn
