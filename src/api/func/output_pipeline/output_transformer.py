# from api.func.reader_pipeline.config_schema import OutputConfig, RuntimeSession
# from typing import Callable, List
# import numpy as np

# def iou(box1, box2):
#     """
#     Calcula el Intersection over Union (IoU) entre dos cajas delimitadoras.
#     El IoU es una medida que indica que tan grande es la superposicion entre dos
#     bounding boxes respecto a su area combinada. Se define como:
#         IoU = Area_de_intersección / Area_de_unión
#     Parametros:
#         box1 (list[float]): Coordenadas [x1, y1, x2, y2] de la primera caja.
#         box2 (list[float]): Coordenadas [x1, y1, x2, y2] de la segunda caja.
#     Retorna:
#         float: Valor entre 0 y 1 que indica el grado de superposicion.
#             0 significa que no se solapan, 1 significa que son identicas.
#     """
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     inter_area = max(0, x2 - x1) * max(0, y2 - y1)
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union_area = box1_area + box2_area - inter_area

#     return inter_area / union_area if union_area != 0 else 0

# def non_max_suppression(predictions: List[List[float]], threshold: float) -> List[List[float]]:
#     """
#     Aplica supresion no maxima (Non-Max Suppression - NMS) para eliminar detecciones redundantes.
#     Proceso:
#         1. Ordena todas las predicciones por su confianza (score) de mayor a menor.
#         2. Itera tomando siempre la prediccion con mayor confianza y la añade a la lista final.
#         3. Elimina de la lista todas las predicciones que tengan un IoU (solapamiento) mayor o igual
#             al `threshold` con la prediccion actual.
#         4. Devuelve unicamente las detecciones seleccionadas.
#     Parametros:
#         - predictions: Lista de cajas con formato [x1, y1, x2, y2, score].
#         - threshold: Umbral de IoU para eliminar cajas solapadas.
#     Retorna:
#         - Lista filtrada de predicciones sin solapamientos significativos.
#     """
#     if not predictions:
#         return []

#     predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
#     keep = []

#     while predictions:
#         current = predictions.pop(0)
#         keep.append(current)
#         predictions = [p for p in predictions if iou(current[:4], p[:4]) < threshold]

#     return keep

# def undo_transform(dets_xyxy: np.ndarray, runtime):
#     """
#     dets_xyxy: np.ndarray shape (N, >=4) en pixeles del TENSOR (xyxy)
#                columnas: [x1, y1, x2, y2, ...]
#     runtime:   RuntimeSession con:
#         - input_width, input_height
#         - orig_width, orig_height
#         - metadata_letter: {scale, pad_left, pad_top, letterbox_used}
#     Devuelve: np.ndarray (N, >=4) en pixeles de la IMAGEN ORIGINAL (xyxy).
#     """

#     x1 = dets_xyxy[:, 0].astype(np.float32, copy=False)
#     y1 = dets_xyxy[:, 1].astype(np.float32, copy=False)
#     x2 = dets_xyxy[:, 2].astype(np.float32, copy=False)
#     y2 = dets_xyxy[:, 3].astype(np.float32, copy=False)

#     W0 = float(runtime.orig_width)
#     H0 = float(runtime.orig_height)

#     md = runtime.metadata_letter or {}
#     letterbox_used = bool(md.get("letterbox_used", False))

#     if letterbox_used:
#         s        = float(md.get("scale", 1.0))
#         pad_left = float(md.get("pad_left", 0.0))
#         pad_top  = float(md.get("pad_top",  0.0))

#         x1 = (x1 - pad_left) / s
#         x2 = (x2 - pad_left) / s
#         y1 = (y1 - pad_top)  / s
#         y2 = (y2 - pad_top)  / s
#     else:
#         W_in = float(runtime.input_width)
#         H_in = float(runtime.input_height)

#         sx = W0 / W_in if W_in > 0 else 1.0
#         sy = H0 / H_in if H_in > 0 else 1.0

#         x1 = x1 * sx; x2 = x2 * sx
#         y1 = y1 * sy; y2 = y2 * sy

#     x1_, x2_ = np.minimum(x1, x2), np.maximum(x1, x2)
#     y1_, y2_ = np.minimum(y1, y2), np.maximum(y1, y2)

#     x1_ = np.clip(x1_, 0.0, W0)
#     x2_ = np.clip(x2_, 0.0, W0)
#     y1_ = np.clip(y1_, 0.0, H0)
#     y2_ = np.clip(y2_, 0.0, H0)

#     out = dets_xyxy.copy()
#     out[:, 0] = x1_; out[:, 1] = y1_
#     out[:, 2] = x2_; out[:, 3] = y2_
#     return out


# def buildPostprocessor(config: OutputConfig, transform_info: RuntimeSession) -> Callable[[List[List[float]]], List[List[float]]]:
#     """
#     Construye una funcion de postprocesamiento que aplica, en orden, los siguientes pasos:
#         1. **Filtrado por confianza**:
#             Elimina todas las detecciones con una confianza menor que `config.confidence_threshold`.
#             Esto reduce la cantidad de cajas a procesar y evita aplicar NMS a detecciones irrelevantes.
#         2. **Supresion no maxima (NMS)** [opcional]:
#             Si `config.apply_nms` es True, elimina detecciones redundantes que tengan un IoU mayor
#             al umbral `config.nms_threshold`.
#         3. **Revertir letterbox** [opcional]:
#             Si `transform_info.metadata_letter["letterbox_used"]` es True, aplica `undo_transform`
#             para devolver las coordenadas al espacio original de la imagen.
#     Parametros:
#         - config: Configuracion con umbral de confianza, uso de NMS y umbral de NMS.
#         - transform_info: Informacion del preprocesamiento que incluye metadatos del letterbox.
#     Retorna:
#         - Una funcion que recibe predicciones (lista de cajas) y devuelve las cajas procesadas.
#     """ 
#     try:
#         steps = []

#         if OutputConfig.output_tensor.output_format != "efficientdet":
#             """
#                 Esto es asi porque 'efficientdet' ya cuenta con un filtro de confianza interno para mejorar 
#             la eficiencia del programa. Aparte es el unico caso en el que no se tiene que usar esta funcion.
#             """
#             steps.append(lambda preds: [p for p in preds if p[4] >= config.confidence_threshold])

#         if config.apply_nms:
#             steps.append(lambda preds: non_max_suppression(preds, config.nms_threshold))

#         if transform_info and transform_info.metadata_letter["letterbox_used"]:
#             steps.append(lambda preds: [undo_transform(p, transform_info.metadata_letter) for p in preds])

#         def postprocess(predictions: List[List[float]]) -> List[List[float]]:
#             for step in steps:
#                 predictions = step(predictions)
#             return predictions

#         return postprocess
#     except Exception as e:
#         raise ValueError(f"Error: {e}") from e
    



# api/func/output_pipeline/output_transformer.py
from __future__ import annotations
from typing import Callable, List, Sequence, Optional
import numpy as np

from api.func.reader_pipeline.config_schema import OutputConfig, RuntimeSession


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """
    NMS clase-agnostico sobre boxes en xyxy.
    boxes: (N,4), scores: (N,)
    Devuelve indices a conservar (ordenados por score desc).
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]

    areas = (np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)).astype(np.float32, copy=False)
    order = np.argsort(scores)[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = (w * h).astype(np.float32, copy=False)

        union = areas[i] + areas[order[1:]] - inter + 1e-12
        iou = inter / union

        remain = np.where(iou <= iou_thr)[0]
        order = order[remain + 1]

    return np.asarray(keep, dtype=np.int64)


def _undo_transform_xyxy_inplace(dets_xyxy: np.ndarray, runtime: RuntimeSession) -> None:
    """
    Modifica IN-PLACE las columnas 0..3 (xyxy) de 'dets_xyxy' para convertir
    de coordenadas en el ESPACIO DEL TENSOR (input_width x input_height)
    a coordenadas en el ESPACIO DE LA IMAGEN ORIGINAL (orig_width x orig_height),
    deshaciendo letterbox si corresponde.
    """
    if dets_xyxy.size == 0:
        return

    boxes = dets_xyxy[:, :4] 

    md = getattr(runtime, "metadata_letter", {}) or {}
    letterbox_usado = bool(md.get("letterbox_used", False))

    if letterbox_usado:
        s  = float(md.get("scale", 1.0))
        pl = float(md.get("pad_left", 0.0))
        pt = float(md.get("pad_top",  0.0))

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pl) / (s + 1e-12)  # x1, x2
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pt) / (s + 1e-12)  # y1, y2
    else:
        W_in = float(getattr(runtime, "input_width",  1) or 1)
        H_in = float(getattr(runtime, "input_height", 1) or 1)
        W0   = float(getattr(runtime, "orig_width",   0))
        H0   = float(getattr(runtime, "orig_height",  0))

        sx = (W0 / W_in) if W_in > 0 else 1.0
        sy = (H0 / H_in) if H_in > 0 else 1.0

        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy

    W0 = float(getattr(runtime, "orig_width",  0))
    H0 = float(getattr(runtime, "orig_height", 0))

    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])

    boxes[:, 0] = np.clip(x1, 0.0, W0)
    boxes[:, 1] = np.clip(y1, 0.0, H0)
    boxes[:, 2] = np.clip(x2, 0.0, W0)
    boxes[:, 3] = np.clip(y2, 0.0, H0)


def buildPostprocessor(output_cfg: OutputConfig, runtime: RuntimeSession) -> Callable[[Sequence[Sequence[float]]], List[List[float]]]:
    """
    Devuelve un callable que toma detecciones en layout:
        [x1, y1, x2, y2, score, class_id]   (floats)  EN EL ESPACIO DEL TENSOR,
    y aplica los pasos finales:

      1) Filtro por confianza (centralizado aqui) — lee el umbral EN CADA LLAMADA.
      2) Top-K opcional (mantener las K mejores por score) antes del NMS.
      3) NMS (clase-agnostico o por clase).
      4) Undo del preprocesado (tensor -> original): letterbox o resize directo.
      5) Orden final por score desc y conversion a List[List[float]].

    Notas:
      - Para modelos con TFLite DetectionPostProcess (ya traen NMS/umbral),
        por defecto desactivamos re-filtrado y re-NMS; se puede forzar via config.
    """
    fmt = (getattr(output_cfg.output_tensor, "output_format", "") or "").lower()
    is_tflite_post = fmt in ("tflite_detpost")

    # ---------parametros-con-defaults-razonables---------
    # Filtro por confianza centralizado (por defecto off para tflite_detpost, on para el resto)
    default_conf_filter = not is_tflite_post
    apply_conf_filter: bool = bool(getattr(output_cfg, "apply_confidence_filter", default_conf_filter))

    # NMS (por defecto off para tflite_detpost, on para el resto)
    default_apply_nms = not is_tflite_post
    apply_nms: bool = bool(getattr(output_cfg, "apply_nms", default_apply_nms))
    iou_thr: float = float(getattr(output_cfg, "iou_threshold", 0.5))
    nms_per_class: bool = bool(getattr(output_cfg, "nms_per_class", False))

    # Top-K opcional
    top_k_opt = getattr(output_cfg, "top_k", None)
    top_k: Optional[int] = int(top_k_opt) if isinstance(top_k_opt, (int, float)) and int(top_k_opt) > 0 else None

    # ---------funcion-de-postproceso---------
    def _postprocess(rows_xyxy: Sequence[Sequence[float]]) -> List[List[float]]:
        if not rows_xyxy:
            return []

        arr = np.asarray(rows_xyxy, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 6:
            raise ValueError(f"output_transformer: se esperaban filas con >=6 columnas (xyxy, score, class). shape={arr.shape}")

        # 1) Filtro por confianza (umbral leido EN CADA LLAMADA -> slider “en vivo”)
        if apply_conf_filter:
            conf_thr = float(getattr(output_cfg, "confidence_threshold", 0.0))
            if conf_thr > 0:
                m = arr[:, 4] >= conf_thr
                if not np.any(m):
                    return []
                arr = arr[m]

        # 2) Top-K (reduce costo del NMS si hay muchas cajas)
        if top_k is not None and arr.shape[0] > top_k:
            order = np.argsort(arr[:, 4])[::-1]
            arr = arr[order[:top_k]]

        # 3) NMS
        if apply_nms and arr.shape[0] > 0:
            if nms_per_class:
                chunks = []
                for cls_val in np.unique(arr[:, 5]):
                    mask = arr[:, 5] == cls_val
                    if not np.any(mask):
                        continue
                    keep = _nms_xyxy(arr[mask, :4], arr[mask, 4], iou_thr)
                    if keep.size:
                        chunks.append(arr[mask][keep])
                arr = np.concatenate(chunks, axis=0) if chunks else arr[:0]
            else:
                keep = _nms_xyxy(arr[:, :4], arr[:, 4], iou_thr)
                arr = arr[keep] if keep.size else arr[:0]

        # 4) Undo del preprocesado → coordenadas en la imagen original
        if arr.shape[0] > 0:
            _undo_transform_xyxy_inplace(arr, runtime)

        # 5) Orden final por score desc (estable para UI/logs)
        if arr.shape[0] > 1:
            arr = arr[np.argsort(arr[:, 4])[::-1]]

        return arr.astype(float, copy=False).tolist()

    return _postprocess