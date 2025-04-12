from .config_schema import InputConfig, OutputConfig
from typing import Callable, List
import numpy as np
import cv2

def buildPreprocessor(config: InputConfig) -> Callable[[np.ndarray], np.ndarray]:
    steps = []
    if config.letterbox:
        pass
        #   steps.append(lambda img: letterboxResize(img, (config.width, config.height)))   <--- ESTA ES LA IDEA CUANDO HAGA LA FUNCION 
    else:
        steps.append(lambda img: cv2.resize(img, (config.width, config.height)))

    if config.scale:
        steps.append(lambda img: img.astype(np.float32) / 255.0)

    if config.normalize:
        mean = np.array(config.mean).reshape(1, 1, -1)
        std = np.array(config.std).reshape(1, 1, -1)
        steps.append(lambda img: (img - mean) / std)

    if config.channels == 1:
        steps.append(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None])

    def preprocess(img):
        for step in steps:
            img = step(img)
        return img

    return preprocess


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

def buildPostprocessor(config: OutputConfig) -> Callable[[List[List[float]]], List[List[float]]]:
    steps = []

    steps.append(lambda preds: [p for p in preds if p[4] >= config.confidence_threshold])

    if config.apply_nms:
        steps.append(lambda preds: non_max_suppression(preds, config.nms_threshold))

    def postprocess(predictions: List[List[float]]) -> List[List[float]]:
        for step in steps:
            predictions = step(predictions)
        return predictions

    return postprocess



'''
    Otra forma de cambiar la confianza sin tener que alterar toda funcion buildPostprocessor seria esta:

def buildPostprocessor(output_config: OutputConfig) -> Callable[[np.ndarray, Optional[float]], Any]:
    def postprocess(output, override_conf=None):
        confidence = override_conf if override_conf is not None else output_config.confidence_threshold
        # Luego usar `confidence` en los filtros...
        return [p for p in output if p[4] >= confidence]
    return postprocess

    No lo veo necesario si se especifica al usuario el no jugar con el cambio de confianza y hacerlo solo cuando sea necesario. 
    Si se lo usa repetidamente se haria inutil el objetivo de "congelar", para mayor rendimiento, la funcion de postProcesamiento para cada IA.
'''