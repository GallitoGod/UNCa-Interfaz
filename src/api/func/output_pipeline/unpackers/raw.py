# # api/func/output_pipeline/unpackers/raw.py
# from __future__ import annotations

# from typing import Any
# import numpy as np
# from .utils import to_2d, rt_shapes

# def build_raw(output_cfg):
#     """
#     Fallback: no interpreta semantica.
#     Devuelve np.ndarray float32 2D para mantener el pipeline estable.

#     Casos:
#       - ndarray/lista 2D -> (N,F)
#       - vector 1D -> (1,F)
#       - tupla/lista de outputs -> (1,K) concatenado (debug)
#     """
#     def fn(raw_output: Any, sh=None) -> np.ndarray:
#         runtime = rt_shapes(sh)
#         if isinstance(raw_output, (list, tuple)) and len(raw_output) > 0 and not np.isscalar(raw_output[0]):
#             try:
#                 flat_parts = []
#                 for t in raw_output:
#                     a = np.asarray(t)
#                     flat_parts.append(a.reshape(-1))
#                 out = np.concatenate(flat_parts, axis=0).astype(np.float32, copy=False)
#                 return out.reshape(1, -1)
#             except Exception:
#                 return np.empty((0, 6), dtype=np.float32)

#         try:
#             arr = to_2d(raw_output).astype(np.float32, copy=False)
#             if arr.ndim == 1:
#                 arr = arr.reshape(1, -1)
#             return arr
#         except Exception:
#             return np.empty((0, 6), dtype=np.float32)

#     return fn


# api/func/output_pipeline/unpackers/raw.py
from __future__ import annotations
from typing import Any
import numpy as np

def build_raw(output_cfg):
    def fn(raw_output: Any, sh=None) -> np.ndarray:
        # ORT: siempre devuelve lista; si hay 1 output, unwrap
        if isinstance(raw_output, (list, tuple)):
            if len(raw_output) == 0:
                return np.empty((0, 6), dtype=np.float32)
            if len(raw_output) == 1:
                raw_output = raw_output[0]
            else:
                # raw no debería “adivinar” múltiples outputs
                raise ValueError(f"raw recibió {len(raw_output)} outputs; no sabe cuál usar.")

        arr = np.asarray(raw_output, dtype=np.float32)

        # Caso típico YOLO: (1, N, F) -> (N, F)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        # Garantizar 2D (N,F)
        if arr.ndim == 1:
            arr = arr[None, :]

        return arr

    return fn