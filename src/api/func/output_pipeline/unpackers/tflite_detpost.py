# api/func/output_pipeline/unpackers/tflite_detpost.py
from __future__ import annotations
import numpy as np
from .utils import to_2d, scale_xyxy_inplace, stack_as_float32_matrix



def build_tflite_detpost(output_cfg):
    """
    **Entrada tipica (NMS ya aplicado en el op TFLite):**
        raw_output = (boxes, scores, classes[, count])
        boxes:   (1,N,4) o (N,4)    [ymin, xmin, ymax, xmax] normalizado
        scores:  (1,N)   o (N,)
        classes: (1,N)   o (N,)
        count:   (1,)    opcional
    Salida:  filas [ymin, xmin, ymax, xmax, score, class_id]
    Nota: no se re-aplica umbral (ya viene filtrado).
    """
    def fn(raw_output, runtime=None):
        boxes, scores, classes = raw_output[0], raw_output[1], raw_output[2]
        count = int(np.asarray(raw_output[3]).reshape(-1)[0]) if len(raw_output) >= 4 else None

        boxes_2d = to_2d(boxes).astype(np.float32, copy=False)
        n_raw = boxes_2d.shape[0]
        N = min(n_raw, count) if count is not None else n_raw

        if N <= 0:
            return np.empty((0, 6), dtype=np.float32)

        ymin, xmin, ymax, xmax = boxes_2d[:N].T
        sc = np.asarray(scores).reshape(-1)[:N].astype(np.float32, copy=False)
        cl = np.asarray(classes).reshape(-1)[:N].astype(np.float32, copy=False)

        # definir bien el contrato de normalizacion con el sistema...
        if runtime is not None and getattr(runtime, "out_coords_space", "tensor_pixels") == "tensor_pixels":
            scale_xyxy_inplace(xmin, ymin, xmax, ymax, (runtime.input_width, runtime.input_height))

        return stack_as_float32_matrix([ymin, xmin, ymax, xmax, sc, cl])
    return fn