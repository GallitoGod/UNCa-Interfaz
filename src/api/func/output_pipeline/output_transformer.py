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