# api/func/output_pipeline/unpackers/build_anchor_deltas.py
from typing import Any, List
import numpy as np
from api.func.reader_pipeline.config_schema import OutputConfig
from .utils import to_2d, decode_anchor_deltas_to_yxyx, stack_as_float32_matrix

'''
    Nota 1 — to_2d en anchor_deltas: cuidado con batch
    Si te llega (1,N,4), to_2d lo convierte a (N,4) perfecto. Bien.
    
    Nota 2 — politica de escalado
    Puse en docstring que anchor_deltas devuelve “PÍXELES DEL TENSOR”.
    Eso esta perfecto, pero mas adelante voy a pasar a lazy scaling, este unpacker va a ser el primero en cambiar a normalized.
'''


def build_anchor_deltas(output_cfg: OutputConfig):
    """
    Entrada cruda (sin DetectionPostProcess):
      raw_output: (box_deltas, class_scores)  o  (class_scores, box_deltas)
        box_deltas:   (1,N,4) o (N,4)  -> [ty, tx, th, tw]
        class_scores: (1,N,C) o (N,C)  -> logits o probas
    Requiere en runtime:
      - anchors (N,4) normalizados [ay, ax, ah, aw]
      - box_variance (4,) típicamente [0.1, 0.1, 0.2, 0.2]
      - input_width/height
    Salida (sin filtrar): [ymin, xmin, ymax, xmax, best_prob, class_id] en PÍXELES DEL TENSOR. <-- Hasta ahora
    """
    def _fn(raw_output: Any, runtime=None) -> np.ndarray:
        if runtime is None or getattr(runtime, "anchors", None) is None:
            raise ValueError("anchor_deltas: falta runtime.anchors (N,4) normalizados.")
        variance = getattr(runtime, "box_variance", None)
        if variance is None:
            variance = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)

        if not isinstance(raw_output, (list, tuple)) or len(raw_output) < 2:
            raise ValueError("anchor_deltas: se espera (box_deltas, class_scores) en una tupla/lista")

        a, b = raw_output[0], raw_output[1]
        A, B = np.asarray(a), np.asarray(b)

        if A.shape[-1] == 4:
            deltas_2d = to_2d(a)      # (N,4)
            cls_2d    = to_2d(b)      # (N,C)
        elif B.shape[-1] == 4:
            deltas_2d = to_2d(b)
            cls_2d    = to_2d(a)
        else:
            raise ValueError("anchor_deltas: no se encontró tensor (N,4) para box_deltas")

        anchors = np.asarray(runtime.anchors, dtype=np.float32)
        if anchors.shape[0] != deltas_2d.shape[0]:
            raise ValueError(f"anchor_deltas: N anchors={anchors.shape[0]} != N deltas={deltas_2d.shape[0]}")

        # 1) activar clases (softmax) y tomar best
        #    (SSD/EfficientDet suele usar softmax multi-clase)
        m = cls_2d - cls_2d.max(axis=1, keepdims=True)
        np.exp(m, out=m)
        cls_prob = m / (m.sum(axis=1, keepdims=True) + 1e-12)

        best_cls = np.argmax(cls_prob, axis=1)
        best_p   = cls_prob[np.arange(cls_prob.shape[0]), best_cls]

        # 2) decodificar deltas -> yxyx NORMALIZADO
        ymin, xmin, ymax, xmax = decode_anchor_deltas_to_yxyx(deltas_2d, anchors, np.asarray(variance))

        # 3) escalar a píxeles del tensor (lo espera el post para undo)
        W, H = int(runtime.input_width), int(runtime.input_height)
        ymin *= H; ymax *= H
        xmin *= W; xmax *= W

        return stack_as_float32_matrix([
            ymin.astype(np.float32, copy=False),
            xmin.astype(np.float32, copy=False),
            ymax.astype(np.float32, copy=False),
            xmax.astype(np.float32, copy=False),
            best_p.astype(np.float32, copy=False),
            best_cls.astype(np.float32, copy=False),
        ])

    return _fn