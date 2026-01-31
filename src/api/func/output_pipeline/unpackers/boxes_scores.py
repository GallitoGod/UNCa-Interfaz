# api/func/output_pipeline/unpackers/boxes_scores.py
from __future__ import annotations
import numpy as np
from .utils import to_2d, scale_xyxy_inplace, stack_as_float32_matrix


def build_boxes_scores(output_cfg):
    def fn(raw_output, runtime=None):
        a, b = raw_output[0], raw_output[1]
        A, B = np.asarray(a), np.asarray(b)

        # Detectar boxes por ultima dimension = 4
        if A.shape[-1] == 4:
            boxes = to_2d(a).astype(np.float32, copy=False)  # (N,4) [ymin,xmin,ymax,xmax]
            scores = to_2d(b).astype(np.float32, copy=False) # (N,C)
        else:
            boxes = to_2d(b).astype(np.float32, copy=False)
            scores = to_2d(a).astype(np.float32, copy=False)

        if boxes.size == 0:
            return np.empty((0, 6), dtype=np.float32)

        best_cls = np.argmax(scores, axis=1).astype(np.int32)
        best_p   = scores[np.arange(scores.shape[0]), best_cls].astype(np.float32, copy=False)

        ymin, xmin, ymax, xmax = boxes.T  # boxes en orden y,x,y,x

        # terminar de definir el contrato de normalizacion con el sistema...
        if runtime is not None and getattr(runtime, "out_coords_space", "tensor_pixels") == "tensor_pixels":
            scale_xyxy_inplace(xmin, ymin, xmax, ymax, (runtime.input_width, runtime.input_height))

        return stack_as_float32_matrix([
            ymin, xmin, ymax, xmax,
            best_p,
            best_cls.astype(np.float32, copy=False)
        ])
    return fn
