# api/func/output_pipeline/unpackers/yolo_flat.py
from __future__ import annotations
import numpy as np
from .utils import to_2d, scale_cxcywh_inplace, stack_as_float32_matrix, rt_shapes

def build_yolo_flat(output_cfg):
    """
    **Entrada:** tensor (N, 5+C) con columnas [cx, cy, w, h, obj, p0..pC]
    Salida:  filas [cx, cy, w, h, score=obj*max_class, class_id]
    Nota: conserva cxcywh (el adapter se encargara de colocarlo correctamente).
    """
    def fn(raw_output, sh=None):
        runtime = rt_shapes(sh)
        arr = to_2d(raw_output).astype(np.float32, copy=False)  # (N, 5+C)
        if arr.size == 0 or arr.shape[1] < 6:
            return np.empty((0, 6), dtype=np.float32)

        cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        obj = arr[:, 4]
        cls = arr[:, 5:]

        best_cls = np.argmax(cls, axis=1).astype(np.int32)
        best_p   = cls[np.arange(cls.shape[0]), best_cls]
        score    = (obj * best_p).astype(np.float32, copy=False)

        # terminar de definir el contrato de normalizacion con el sistema...
        if runtime is not None and getattr(runtime, "out_coords_space", "tensor_pixels") == "tensor_pixels":
            scale_cxcywh_inplace(cx, cy, w, h, (runtime.input_width, runtime.input_height))

        return stack_as_float32_matrix([
            cx, cy, w, h,
            score,
            best_cls.astype(np.float32, copy=False)
        ])
    return fn