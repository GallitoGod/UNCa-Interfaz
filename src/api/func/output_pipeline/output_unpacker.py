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

# ------------------------Callbacks------------------------

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
