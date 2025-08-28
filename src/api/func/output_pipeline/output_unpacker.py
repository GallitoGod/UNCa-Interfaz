# api/func/output_pipeline/output_unpacker.py
# from api.func.reader_pipeline.config_schema import OutputConfig
# from typing import Any, Callable, List, Union
# import numpy as np

# def unpack_out(output: OutputConfig) -> Callable[[Any], Union[List[List[float]], np.ndarray]]:
#     """
#     Devuelve una funcion especializada para desempaquetar el raw_output de un modelo
#     segun el formato declarado en output_tensor.output_format.
    
#     La funcion retornada convierte el output crudo en una estructura estandarizada
#     para el adaptador de salida: List[List[float]] o np.ndarray.
    
#     :param output_format: 'raw', 'multihead', 'softmax' o 'efficientdet'.
#     :return: Funcion desempaquetadora especializada
#     """

#     if output.output_tensor.output_format == "raw":
#         return lambda r: r

#     elif output.output_tensor.output_format == "multihead":     
#         #   <- Cada feature map produce una deteccion independiente, por ello se terminan concatenando
#         return lambda r: np.concatenate([np.array(x) for x in r], axis=-1)

#     elif output.output_tensor.output_format == "softmax":       
#         #   <- No es muy relevante en deteccion, terminara en clasificacion
#         def unpack_softmax(raw_output):
#             arr = np.array(raw_output)
#             if arr.ndim == 1:
#                 arr = arr.reshape(1, -1)
#             results = []
#             for row in arr:
#                 class_id = int(np.argmax(row))
#                 confidence = float(np.max(row))
#                 results.append([0.0, 0.0, 0.0, 0.0, confidence, class_id])
#             return results
#         return unpack_softmax
    
#     elif output.output_tensor.output_format == "efficientdet":
#         def parse_efficientdet(raw_output, image_shape=None):
#             """
#             raw_output admite:
#             - (boxes, class_scores[, ...])  ó
#             - (class_scores, boxes[, ...])
#             Donde:
#             boxes:       (1, N, 4) en orden [ymin, xmin, ymax, xmax], normalizadas [0..1]
#             class_scores:(1, N, C) probabilidades o logits ya softmaxeadas

#             image_shape: (W, H) del tensor de entrada al modelo.

#             Devuelve: List[List[float]] con [x1, y1, x2, y2, score, class_id] en floats.
#             """
#             a, b = raw_output[0], raw_output[1]

#             # Detectar cual es boxes por el ultimo eje
#             if a.shape[-1] == 4:
#                 boxes = a         # (1, N, 4)
#                 class_scores = b  # (1, N, C)
#             else:
#                 class_scores = a
#                 boxes = b

#             scores = class_scores[0]      # (N, C)
#             bxy = boxes[0]                # (N, 4) [ymin,xmin,ymax,xmax]

#             # Mejor clase por anchor (vectorizado) numpy
#             best_cls = np.argmax(scores, axis=1)                            # (N,)
#             best_sc = scores[np.arange(scores.shape[0]), best_cls]          # (N,)

#             # Filtro por umbral (vectorizado) numpy
#             mask = best_sc >= float(output.confidence_threshold)
#             if not np.any(mask):
#                 return []

#             sel = bxy[mask]                          # (K, 4)
#             sc  = best_sc[mask].astype(np.float32)   # (K,)
#             cl  = best_cls[mask].astype(np.float32)  # float por contrato del sistema

#             ymin, xmin, ymax, xmax = sel.T
#             x1, y1, x2, y2 = xmin, ymin, xmax, ymax  # a [x1,y1,x2,y2]

#             # Escalado a pixeles del tensor de entrada
#             if image_shape is not None:
#                 W, H = image_shape  
#                 x1 = x1 * W; x2 = x2 * W
#                 y1 = y1 * H; y2 = y2 * H

#             dets = np.column_stack([x1, y1, x2, y2, sc, cl]).astype(float)

#             return dets.tolist()
#         return parse_efficientdet

#     else:
#         raise ValueError(f"output_format desconocido: {output.output_tensor.output_format}")

#   NUEVO UNPACKER  ------>>
#   Lo cambio porque este soporta mejor los cambios entre salidas de distintas inferencias y esta totalemte vectorizado.
#   Aparte, la idea es normalizar los boxes para que undo_transform en el posprocess funcione correctamente.

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

def unpack_yolo_flat(output_cfg: OutputConfig) -> Callable[[Any, Optional[ImageSize]], List[List[float]]]:
    """
    **Entrada:** tensor (N, 5+C) con columnas [cx, cy, w, h, obj, p0..pC]
    Salida:  filas [cx, cy, w, h, score=obj*max_class, class_id]
    Nota: conserva cxcywh (el adapter se encargara de colocarlo correctamente).
    """
    confidence_thr = float(output_cfg.confidence_threshold)

    def _fn(raw_output: Any, input_tensor_size: Optional[ImageSize] = None) -> List[List[float]]:
        preds_2d = _to_2d(raw_output)  # (N, 5+C)
        if preds_2d.ndim != 2 or preds_2d.shape[1] < 6:
            raise ValueError(f"yolo_flat: shape inválido {preds_2d.shape}; se espera (N, 5+C)")

        cx, cy, w, h = preds_2d[:, 0], preds_2d[:, 1], preds_2d[:, 2], preds_2d[:, 3]
        objectness = preds_2d[:, 4]
        class_scores_or_logits = preds_2d[:, 5:]

        best_class_indices = np.argmax(class_scores_or_logits, axis=1)
        best_class_scores  = class_scores_or_logits[np.arange(class_scores_or_logits.shape[0]), best_class_indices]

        combined_score = objectness * best_class_scores
        valid_mask = combined_score >= confidence_thr
        if not np.any(valid_mask):
            return []

        cx, cy, w, h = cx[valid_mask], cy[valid_mask], w[valid_mask], h[valid_mask]
        scores = combined_score[valid_mask].astype(np.float32, copy=False)
        class_ids = best_class_indices[valid_mask].astype(np.float32, copy=False)

        # Escala a pixeles del tensor de entrada si parece normalizado y se pasa (W,H)
        if input_tensor_size is not None and _looks_normalized_cxcywh(cx, cy, w, h):
            _scale_cxcywh_inplace(cx, cy, w, h, input_tensor_size)

        return _stack_as_float_lists([cx, cy, w, h, scores, class_ids])

    return _fn


def unpack_boxes_scores(output_cfg: OutputConfig) -> Callable[[Any, Optional[ImageSize]], List[List[float]]]:
    """
    **Entrada:** (boxes, class_scores) en cualquier orden. Shapes tipicos:
        boxes: (1,N,4) o (N,4)   en [ymin, xmin, ymax, xmax], o lo que sea
        class_scores: (1,N,C) o (N,C)
    Salida:  filas [ymin, xmin, ymax, xmax, max_class_score, class_id]...
    Nota: conserva yxyx, o lo que sea, (el adapter se encarga de la conversion de vectores).
    """
    confidence_thr = float(output_cfg.confidence_threshold)

    def _fn(raw_output: Any, input_tensor_size: Optional[ImageSize] = None) -> List[List[float]]:
        if not isinstance(raw_output, (list, tuple)) or len(raw_output) < 2:
            raise ValueError("boxes_scores: se espera (boxes, class_scores) en una tupla/lista")

        first, second = raw_output[0], raw_output[1]
        first_arr, second_arr = np.asarray(first), np.asarray(second)

        # Deteccion de boxes por ultima dimension = 4
        if first_arr.shape[-1] == 4:
            boxes_2d = _to_2d(first)      # (N,4)  [ymin, xmin, ymax, xmax]
            scores_2d = _to_2d(second)    # (N,C)
        elif second_arr.shape[-1] == 4:
            boxes_2d = _to_2d(second)
            scores_2d = _to_2d(first)
        else:
            raise ValueError("boxes_scores: no se encontro tensor de cajas con ultima dimension = 4")

        if boxes_2d.shape[0] != scores_2d.shape[0]:
            raise ValueError(f"boxes_scores: N inconsistente entre boxes {boxes_2d.shape} y scores {scores_2d.shape}")

        best_class_indices = np.argmax(scores_2d, axis=1)   # Esto es una vectorizacion
        best_class_scores  = scores_2d[np.arange(scores_2d.shape[0]), best_class_indices]

        valid_mask = best_class_scores >= confidence_thr
        if not np.any(valid_mask):
            return []

        ymin, xmin, ymax, xmax = boxes_2d[valid_mask].T
        scores   = best_class_scores[valid_mask].astype(np.float32, copy=False)
        class_ids = best_class_indices[valid_mask].astype(np.float32, copy=False)

        if input_tensor_size is not None and _looks_normalized_xyxy(xmin, ymin, xmax, ymax):
            _scale_xyxy_inplace(xmin, ymin, xmax, ymax, input_tensor_size)

        return _stack_as_float_lists([ymin, xmin, ymax, xmax, scores, class_ids])

    return _fn


def unpack_tflite_detpost(output_cfg: OutputConfig) -> Callable[[Any, Optional[ImageSize]], List[List[float]]]:
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
    def _fn(raw_output: Any, input_tensor_size: Optional[ImageSize] = None) -> List[List[float]]:
        if not isinstance(raw_output, (list, tuple)) or len(raw_output) < 3:
            raise ValueError("tflite_detpost: se espera (boxes, scores, classes[, count])")

        # Se asume un orden tipico. En caso de no ser este el orden, se podria alterar esta funcion,
        # aunque no suele ser el caso y tampoco quiero hacerlo.
        boxes, scores, classes = raw_output[0], raw_output[1], raw_output[2]
        count = None
        if len(raw_output) >= 4:
            count = int(np.asarray(raw_output[3]).reshape(-1)[0])

        boxes_2d   = _to_2d(boxes)
        scores_1d  = np.asarray(scores).reshape(-1)
        classes_1d = np.asarray(classes).reshape(-1)

        N = boxes_2d.shape[0]
        if count is not None:
            N = min(N, count)

        ymin, xmin, ymax, xmax = boxes_2d[:N].T
        scores_slice   = scores_1d[:N].astype(np.float32, copy=False)
        class_ids_slice = classes_1d[:N].astype(np.float32, copy=False)

        if input_tensor_size is not None and _looks_normalized_xyxy(xmin, ymin, xmax, ymax):
            _scale_xyxy_inplace(xmin, ymin, xmax, ymax, input_tensor_size)

        return _stack_as_float_lists([ymin, xmin, ymax, xmax, scores_slice, class_ids_slice])

    return _fn

# ------------------------Cabeza de unpacker------------------------

def unpack_out(output_cfg: OutputConfig) -> Callable[[Any, Optional[ImageSize]], List[List[float]]]:
    """
    **Devuelve un callback liviano que transforma la salida cruda del modelo en filas:**
        [b0, b1, b2, b3, score, class_id]  (floats)
    donde b0..b3 permanecen en el formato nativo de la IA (YOLO: cx,cy,w,h; SSD/EfficientDet: ymin,xmin,ymax,xmax).
    El adapter posterior convierte a xyxy 'tensor_structure'.
    """
    fmt = (output_cfg.output_tensor.output_format or "").lower()

    if fmt in ("yolo_flat"):
        return unpack_yolo_flat(output_cfg)

    if fmt in ("boxes_scores"):
        return unpack_boxes_scores(output_cfg)

    if fmt in ("tflite_detpost"):
        return unpack_tflite_detpost(output_cfg)

    if fmt == "raw":
        # Para pruebas o modelos que ya devuelven el contrato final.
        return lambda raw_output, input_tensor_size=None: raw_output

    raise ValueError(
        f"output_format no soportado: '{fmt}'. Usa uno de: "
        f"yolo_flat | boxes_scores | tflite_detpost | raw "
    )
