# api/func/output_pipeline/unpackers/raw.py
from __future__ import annotations

from typing import Any
import numpy as np
from .utils import to_2d, rt_shapes

def build_raw(output_cfg):
    """
    Fallback: no interpreta semantica.
    Devuelve np.ndarray float32 2D para mantener el pipeline estable.

    Casos:
      - ndarray/lista 2D -> (N,F)
      - vector 1D -> (1,F)
      - tupla/lista de outputs -> (1,K) concatenado (debug)
    """
    def fn(raw_output: Any, sh=None) -> np.ndarray:
        runtime = rt_shapes(sh)
        # Caso: multiples outputs
        if isinstance(raw_output, (list, tuple)) and len(raw_output) > 0 and not np.isscalar(raw_output[0]):
            try:
                flat_parts = []
                for t in raw_output:
                    a = np.asarray(t)
                    flat_parts.append(a.reshape(-1))
                out = np.concatenate(flat_parts, axis=0).astype(np.float32, copy=False)
                return out.reshape(1, -1)
            except Exception:
                # fallback duro
                return np.empty((0, 6), dtype=np.float32)

        # Caso: output simple
        try:
            arr = to_2d(raw_output).astype(np.float32, copy=False)
            # Garantizar 2D
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception:
            return np.empty((0, 6), dtype=np.float32)

    return fn
