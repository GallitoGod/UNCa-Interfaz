# api/func/output_pipeline/output_unpacker.py
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
from api.func.reader_pipeline.config_schema import OutputConfig

NDArray = np.ndarray
ImageSize = Tuple[int, int]  # (W, H)

# ------------------------Utiles------------------------

def _to_2d(array_like: Any) -> NDArray:
    """
    Convierte a np.ndarray y normaliza a 2D:
      - (1, N, F) -> (N, F)
      - (N,) -> (N, 1)
      - (N, A, B, ...) -> (N, A*B*...)
    """
    arr = np.asarray(array_like)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

def _looks_normalized_xyxy(x1: NDArray, y1: NDArray, x2: NDArray, y2: NDArray) -> bool:
    """
    si los maximos estan cerca de 1, se asume [0..1], por ende, normalizados.
    """
    max_val = float(np.max([x1.max(initial=0.0), y1.max(initial=0.0),
                            x2.max(initial=0.0), y2.max(initial=0.0)]))
    return max_val <= 1.5

def _looks_normalized_cxcywh(cx: NDArray, cy: NDArray, w: NDArray, h: NDArray) -> bool:
    max_val = float(np.max([cx.max(initial=0.0), cy.max(initial=0.0),
                            w.max(initial=0.0),  h.max(initial=0.0)]))
    return max_val <= 1.5

def _scale_xyxy_inplace(x1: NDArray, y1: NDArray, x2: NDArray, y2: NDArray,
                        input_tensor_size: ImageSize) -> None:
    W, H = input_tensor_size
    x1 *= W; x2 *= W
    y1 *= H; y2 *= H

def _scale_cxcywh_inplace(cx: NDArray, cy: NDArray, w: NDArray, h: NDArray,
                          input_tensor_size: ImageSize) -> None:
    W, H = input_tensor_size
    cx *= W; w *= W
    cy *= H; h *= H

def _stack_as_float_lists(columns: List[NDArray]) -> List[List[float]]:
    """
    Apila columnas 1D → matriz 2D y la devuelve como List[List[float]].
    """
    mat = np.column_stack(columns).astype(float, copy=False)
    return mat.tolist()

def _decode_anchor_deltas_to_yxyx(
    deltas: NDArray, anchors: NDArray, variance: NDArray
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    deltas:  (N,4) -> [ty, tx, th, tw]
    anchors: (N,4) -> [ay, ax, ah, aw]   (centro y tamaño), NORMALIZADOS [0..1]
    variance: (4,) -> [sy, sx, sh, sw]
    Devuelve: (ymin, xmin, ymax, xmax) NORMALIZADOS [0..1]
    """
    ty, tx, th, tw = deltas.T
    ay, ax, ah, aw = anchors.T
    sy, sx, sh, sw = variance.astype(np.float32)

    yc = ty * sy * ah + ay
    xc = tx * sx * aw + ax
    h  = np.exp(th * sh) * ah
    w  = np.exp(tw * sw) * aw

    ymin = yc - 0.5 * h
    xmin = xc - 0.5 * w
    ymax = yc + 0.5 * h
    xmax = xc + 0.5 * w
    return ymin, xmin, ymax, xmax
    

# ------------------------Callbacks------------------------

def _unpack_anchor_deltas(output_cfg: OutputConfig):
    """
    Entrada cruda (sin DetectionPostProcess):
      raw_output: (box_deltas, class_scores)  o  (class_scores, box_deltas)
        box_deltas:   (1,N,4) o (N,4)  -> [ty, tx, th, tw]
        class_scores: (1,N,C) o (N,C)  -> logits o probas
    Requiere en runtime:
      - anchors (N,4) normalizados [ay, ax, ah, aw]
      - box_variance (4,) típicamente [0.1, 0.1, 0.2, 0.2]
      - input_width/height
    Salida (sin filtrar): [ymin, xmin, ymax, xmax, best_prob, class_id] en PÍXELES DEL TENSOR.
    """
    def _fn(raw_output: Any, runtime=None) -> List[List[float]]:
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
            deltas_2d = _to_2d(a)      # (N,4)
            cls_2d    = _to_2d(b)      # (N,C)
        elif B.shape[-1] == 4:
            deltas_2d = _to_2d(b)
            cls_2d    = _to_2d(a)
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
        ymin, xmin, ymax, xmax = _decode_anchor_deltas_to_yxyx(deltas_2d, anchors, np.asarray(variance))

        # 3) escalar a píxeles del tensor (lo espera el post para undo)
        W, H = int(runtime.input_width), int(runtime.input_height)
        ymin *= H; ymax *= H
        xmin *= W; xmax *= W

        return _stack_as_float_lists([
            ymin.astype(np.float32, copy=False),
            xmin.astype(np.float32, copy=False),
            ymax.astype(np.float32, copy=False),
            xmax.astype(np.float32, copy=False),
            best_p.astype(np.float32, copy=False),
            best_cls.astype(np.float32, copy=False),
        ])

    return _fn


def _unpack_yolo_flat(output_cfg):
    """
    **Entrada:** tensor (N, 5+C) con columnas [cx, cy, w, h, obj, p0..pC]
    Salida:  filas [cx, cy, w, h, score=obj*max_class, class_id]
    Nota: conserva cxcywh (el adapter se encargara de colocarlo correctamente).
    """
    def fn(raw_output, runtime=None):
        arr = _to_2d(raw_output).astype(np.float32, copy=True)  # (N, 5+C)
        cx, cy, w, h = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
        obj = arr[:,4]
        cls = arr[:,5:]
        best_cls = np.argmax(cls, axis=1)
        best_p   = cls[np.arange(cls.shape[0]), best_cls]
        score    = obj * best_p

        if runtime is not None and getattr(runtime, "out_coords_space", "tensor_pixels") == "normalized_0_1":
            _scale_cxcywh_inplace(cx, cy, w, h, (runtime.input_width, runtime.input_height))

        return _stack_as_float_lists([cx, cy, w, h, score.astype(np.float32), best_cls.astype(np.float32)])
    return fn


def _unpack_boxes_scores(output_cfg):
    def fn(raw_output, runtime=None):
        a,b = raw_output[0], raw_output[1]
        A,B = np.asarray(a), np.asarray(b)
        boxes = (_to_2d(a) if A.shape[-1]==4 else _to_2d(b)).astype(np.float32, copy=True)     # [ymin,xmin,ymax,xmax]
        scores= _to_2d(b) if A.shape[-1]==4 else _to_2d(a)     # (N,C)
        best_cls = np.argmax(scores, axis=1)
        best_p   = scores[np.arange(scores.shape[0]), best_cls]
        ymin, xmin, ymax, xmax = boxes.T

        if runtime is not None and getattr(runtime, "out_coords_space", "tensor_pixels") == "normalized_0_1":
            _scale_xyxy_inplace(xmin, ymin, xmax, ymax, (runtime.input_width, runtime.input_height))

        return _stack_as_float_lists([ymin, xmin, ymax, xmax, best_p.astype(np.float32), best_cls.astype(np.float32)])
    return fn


def _unpack_tflite_detpost(output_cfg):
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
        count = int(np.asarray(raw_output[3]).reshape(-1)[0]) if len(raw_output)>=4 else None
        boxes_2d = _to_2d(boxes).astype(np.float32, copy=True)
        N = min(boxes_2d.shape[0], count) if count is not None else boxes_2d.shape[0]
        ymin, xmin, ymax, xmax = boxes_2d[:N].T
        sc = np.asarray(scores).reshape(-1)[:N].astype(np.float32)
        cl = np.asarray(classes).reshape(-1)[:N].astype(np.float32)

        if runtime is not None and getattr(runtime, "out_coords_space", "tensor_pixels") == "normalized_0_1":
            _scale_xyxy_inplace(xmin, ymin, xmax, ymax, (runtime.input_width, runtime.input_height))

        return _stack_as_float_lists([ymin, xmin, ymax, xmax, sc, cl])
    return fn

# ------------------------Cabeza de unpacker------------------------

def unpack_out(output_cfg: OutputConfig) -> Callable[[Any, Optional[ImageSize]], List[List[float]]]:
    """
    **Devuelve un callback que transforma la salida cruda del modelo en filas:**
        [b0, b1, b2, b3, score, class_id]  (floats)
    donde b0..b3 permanecen en el formato nativo de la IA (YOLO: cx,cy,w,h; SSD/EfficientDet: ymin,xmin,ymax,xmax).
    El adapter posterior convierte a xyxy 'tensor_structure'.
    """
    fmt = (output_cfg.pack_format or "").lower()

    if fmt in ("yolo_flat"):
        return _unpack_yolo_flat(output_cfg)
    
    if fmt in ("anchor_deltas"):
        return _decode_anchor_deltas_to_yxyx(output_cfg)

    if fmt in ("boxes_scores"):
        return _unpack_boxes_scores(output_cfg)

    if fmt in ("tflite_detpost"):
        return _unpack_tflite_detpost(output_cfg)

    if fmt == "raw":
        return lambda raw_output, runTime : raw_output

    raise ValueError(
        f"output_format no soportado: '{fmt}'. Usa uno de: "
        f"yolo_flat | boxes_scores | tflite_detpost | raw "
    )
