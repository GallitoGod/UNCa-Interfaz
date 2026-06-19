# api/func/output_pipeline/unpackers/_shape.py
from __future__ import annotations
from typing import Any
import numpy as np


def normalize_to_2d(arr: Any) -> np.ndarray:
    """Contrato UNICO de forma de la capa de unpackers.

    Deja la salida cruda de cualquier unpacker como un ndarray 2D float32 (N, K).
    Aca converge TODA la normalizacion de shapes que antes estaba DUPLICADA entre
    raw.py y model_controller.inference() (reforma Fase 2, tarea 2). El controller
    ya no toca shapes: confia en que el unpacker entrega (N, K) float32.

    Reglas (las mismas que aplicaban raw.py y el controller, unificadas):
      - list/tuple vacio                -> (0, 6) float32 (sin detecciones)
      - list/tuple de 1 elemento        -> se desempaqueta ese elemento
      - list/tuple de >1 elemento       -> error: varios tensores, desempaque ambiguo
      - ndarray 3D con batch 1 (1,N,K)  -> se aplasta a (N, K)
      - ndarray 1D (K,)                 -> se expande a (1, K)
    """
    if isinstance(arr, (list, tuple)):
        if len(arr) == 0:
            return np.empty((0, 6), dtype=np.float32)
        if len(arr) == 1:
            arr = arr[0]
        else:
            raise ValueError(
                f"normalize_to_2d: se recibieron {len(arr)} outputs; "
                "desempaque ambiguo (no se sabe cual es la matriz de detecciones).")

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.ndim != 2:
        raise ValueError(
            f"normalize_to_2d: no se pudo llevar la salida a 2D (shape final {arr.shape}).")

    return arr
