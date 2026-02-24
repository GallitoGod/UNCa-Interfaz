# api/func/output_pipeline/unpackers/raw.py
from __future__ import annotations
from typing import Any
import numpy as np

def build_raw(output_cfg):
    def fn(raw_output: Any, sh=None) -> np.ndarray:
        if isinstance(raw_output, (list, tuple)):
            if len(raw_output) == 0:
                return np.empty((0, 6), dtype=np.float32)
            if len(raw_output) == 1:
                raw_output = raw_output[0]
            else:
                raise ValueError(f"raw recibio {len(raw_output)} outputs; no sabe cual usar.")

        arr = np.asarray(raw_output, dtype=np.float32)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim == 1:
            arr = arr[None, :]

        return arr

    return fn