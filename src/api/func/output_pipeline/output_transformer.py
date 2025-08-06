from ..reader_pipeline import OutputConfig, RuntimeSession
from typing import Callable, List

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

def buildPostprocessor(config: OutputConfig, transform_info: RuntimeSession) -> Callable[[List[List[float]]], List[List[float]]]:
    #El orden aqui siempre debe ser: 1_Filtrar por confianza 2_Aplicar nms 3_Deshacer letterbox(Si es que se aplico). 
    try:
        steps = []

        steps.append(lambda preds: [p for p in preds if p[4] >= config.confidence_threshold])

        if config.apply_nms:
            steps.append(lambda preds: non_max_suppression(preds, config.nms_threshold))

        if transform_info and transform_info.metadata_letter["letterbox_used"]:
            steps.append(lambda preds: [undo_transform(p, transform_info.metadata_letter) for p in preds])


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