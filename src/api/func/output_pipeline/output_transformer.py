from api.func.reader_pipeline.config_schema import OutputConfig, RuntimeSession
from typing import Callable, List
import numpy as np

def iou(box1, box2):
    """
    Calcula el Intersection over Union (IoU) entre dos cajas delimitadoras.
    El IoU es una medida que indica que tan grande es la superposicion entre dos
    bounding boxes respecto a su area combinada. Se define como:
        IoU = Area_de_intersección / Area_de_unión
    Parametros:
        box1 (list[float]): Coordenadas [x1, y1, x2, y2] de la primera caja.
        box2 (list[float]): Coordenadas [x1, y1, x2, y2] de la segunda caja.
    Retorna:
        float: Valor entre 0 y 1 que indica el grado de superposicion.
            0 significa que no se solapan, 1 significa que son identicas.
    """
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
    """
    Aplica supresion no maxima (Non-Max Suppression - NMS) para eliminar detecciones redundantes.
    Proceso:
        1. Ordena todas las predicciones por su confianza (score) de mayor a menor.
        2. Itera tomando siempre la prediccion con mayor confianza y la añade a la lista final.
        3. Elimina de la lista todas las predicciones que tengan un IoU (solapamiento) mayor o igual
            al `threshold` con la prediccion actual.
        4. Devuelve unicamente las detecciones seleccionadas.
    Parametros:
        - predictions: Lista de cajas con formato [x1, y1, x2, y2, score].
        - threshold: Umbral de IoU para eliminar cajas solapadas.
    Retorna:
        - Lista filtrada de predicciones sin solapamientos significativos.
    """
    if not predictions:
        return []

    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    keep = []

    while predictions:
        current = predictions.pop(0)
        keep.append(current)
        predictions = [p for p in predictions if iou(current[:4], p[:4]) < threshold]

    return keep

def undo_transform(dets_xyxy: np.ndarray, runtime):
    """
    dets_xyxy: np.ndarray shape (N, >=4) en pixeles del TENSOR (xyxy)
               columnas: [x1, y1, x2, y2, ...]
    runtime:   RuntimeSession con:
        - input_width, input_height
        - orig_width, orig_height
        - metadata_letter: {scale, pad_left, pad_top, letterbox_used}
    Devuelve: np.ndarray (N, >=4) en pixeles de la IMAGEN ORIGINAL (xyxy).
    """

    x1 = dets_xyxy[:, 0].astype(np.float32, copy=False)
    y1 = dets_xyxy[:, 1].astype(np.float32, copy=False)
    x2 = dets_xyxy[:, 2].astype(np.float32, copy=False)
    y2 = dets_xyxy[:, 3].astype(np.float32, copy=False)

    W0 = float(runtime.orig_width)
    H0 = float(runtime.orig_height)

    md = runtime.metadata_letter or {}
    letterbox_used = bool(md.get("letterbox_used", False))

    if letterbox_used:
        s        = float(md.get("scale", 1.0))
        pad_left = float(md.get("pad_left", 0.0))
        pad_top  = float(md.get("pad_top",  0.0))

        x1 = (x1 - pad_left) / s
        x2 = (x2 - pad_left) / s
        y1 = (y1 - pad_top)  / s
        y2 = (y2 - pad_top)  / s
    else:
        W_in = float(runtime.input_width)
        H_in = float(runtime.input_height)

        sx = W0 / W_in if W_in > 0 else 1.0
        sy = H0 / H_in if H_in > 0 else 1.0

        x1 = x1 * sx; x2 = x2 * sx
        y1 = y1 * sy; y2 = y2 * sy

    x1_, x2_ = np.minimum(x1, x2), np.maximum(x1, x2)
    y1_, y2_ = np.minimum(y1, y2), np.maximum(y1, y2)

    x1_ = np.clip(x1_, 0.0, W0)
    x2_ = np.clip(x2_, 0.0, W0)
    y1_ = np.clip(y1_, 0.0, H0)
    y2_ = np.clip(y2_, 0.0, H0)

    out = dets_xyxy.copy()
    out[:, 0] = x1_; out[:, 1] = y1_
    out[:, 2] = x2_; out[:, 3] = y2_
    return out


def buildPostprocessor(config: OutputConfig, transform_info: RuntimeSession) -> Callable[[List[List[float]]], List[List[float]]]:
    """
    Construye una funcion de postprocesamiento que aplica, en orden, los siguientes pasos:
        1. **Filtrado por confianza**:
            Elimina todas las detecciones con una confianza menor que `config.confidence_threshold`.
            Esto reduce la cantidad de cajas a procesar y evita aplicar NMS a detecciones irrelevantes.
        2. **Supresion no maxima (NMS)** [opcional]:
            Si `config.apply_nms` es True, elimina detecciones redundantes que tengan un IoU mayor
            al umbral `config.nms_threshold`.
        3. **Revertir letterbox** [opcional]:
            Si `transform_info.metadata_letter["letterbox_used"]` es True, aplica `undo_transform`
            para devolver las coordenadas al espacio original de la imagen.
    Parametros:
        - config: Configuracion con umbral de confianza, uso de NMS y umbral de NMS.
        - transform_info: Informacion del preprocesamiento que incluye metadatos del letterbox.
    Retorna:
        - Una funcion que recibe predicciones (lista de cajas) y devuelve las cajas procesadas.
    """ 
    try:
        steps = []

        if OutputConfig.output_tensor.output_format != "efficientdet":
            """
                Esto es asi porque 'efficientdet' ya cuenta con un filtro de confianza interno para mejorar 
            la eficiencia del programa. Aparte es el unico caso en el que no se tiene que usar esta funcion.
            """
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