from .config_schema import InputConfig, OutputConfig
from typing import Callable, List
import numpy as np
import cv2

def build_letterbox(input_width, input_height, pad_color):
    def letterbox(img):
        h, w = img.shape[:2]
        scale = min(input_width / w, input_height / h)
        nw, nh = int(w * scale), int(h * scale)
        pad_w = input_width - nw
        pad_h = input_height - nh
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        resized = cv2.resize(img, (nw, nh))
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color
        )
        return padded, scale, pad_left, pad_top

    return letterbox

def buildPreprocessor(config: InputConfig) -> Callable[[np.ndarray], np.ndarray]:
    try:
        steps = []
        transform_info = {}
        if config.letterbox:
            letterbox = build_letterbox(config.width, config.height, config.auto_pad_color)

            def letterbox_wrapper(img):
                padded, scale, pad_left, pad_top = letterbox(img)
                transform_info['scale'] = scale
                transform_info['pad_left'] = pad_left
                transform_info['pad_top'] = pad_top
                transform_info['used_letterbox'] = True
                return padded

            steps.append(letterbox_wrapper)
        else:
            steps.append(lambda img: cv2.resize(img, (config.width, config.height)))

        if config.scale:
            steps.append(lambda img: img.astype(np.float32) / 255.0)

        if config.normalize:
            mean = np.array(config.mean).reshape(1, 1, -1)
            std = np.array(config.std).reshape(1, 1, -1)
            steps.append(lambda img: (img - mean) / std)

        def preprocess(img):
            for step in steps:
                img = step(img)
            return img

        return preprocess, transform_info
    except Exception as e:
        raise ValueError(f"Error: {e}") from e


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def non_max_suppression(predictions: List[List[float]], threshold: float) -> List[List[float]]:
    if not predictions:
        return []

    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    keep = []

    while predictions:
        current = predictions.pop(0)
        keep.append(current)
        predictions = [p for p in predictions if iou(current[:4], p[:4]) < threshold]

    return keep

def undo_transform(p, transform_info):
    scale = transform_info["scale"]
    pad_left = transform_info["pad_left"]
    pad_top = transform_info["pad_top"]
    p[0] = (p[0] - pad_left) / scale
    p[1] = (p[1] - pad_top) / scale
    p[2] = (p[2] - pad_left) / scale
    p[3] = (p[3] - pad_top) / scale
    return p

def buildPostprocessor(config: OutputConfig, transform_info: dict = None) -> Callable[[List[List[float]]], List[List[float]]]:
    #El orden aqui siempre debe ser: 1_Filtrar por confianza 2_Aplicar nms 3_Deshacer letterbox(Si es que se aplico). 
    try:
        steps = []

        steps.append(lambda preds: [p for p in preds if p[4] >= config.confidence_threshold])

        if config.apply_nms:
            steps.append(lambda preds: non_max_suppression(preds, config.nms_threshold))

        if transform_info and transform_info.get("used_letterbox", True):
            steps.append(lambda preds: [undo_transform(p, transform_info) for p in preds])


        def postprocess(predictions: List[List[float]]) -> List[List[float]]:
            for step in steps:
                predictions = step(predictions)
            return predictions

        return postprocess
    except Exception as e:
        raise ValueError(f"Error: {e}") from e


'''
    Ahora la confianza si cambia con el cambio desde el cliente, el transformador pasa el valor de la confianza 
al NMS a traves de el valor mutable "nms_threshold" que toma valor de "ReactiveOutputConfig". Ya no se necesita
descongelar "postprocess" ya que no toma un valor estatico sino que lee el ultimo valor de confianza antes de 
hacer funcionar NMS.
'''