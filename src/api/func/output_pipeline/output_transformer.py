from api.func.reader_pipeline.config_schema import OutputConfig, RuntimeSession
from typing import Callable, List

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

def undo_transform(p, transform_info):
    """
    Revierte las transformaciones de escala y padding aplicadas durante el preprocesamiento
    (por ejemplo, las que genera el letterbox).
    Proceso:
        1. Resta el padding izquierdo y superior a las coordenadas x1, y1, x2, y2.
        2. Divide las coordenadas resultantes por el factor de escala original.
        3. Devuelve las coordenadas restauradas al espacio original de la imagen.
    Parametros:
        - p: Lista [x1, y1, x2, y2, ...] que representa una caja detectada.
        - transform_info: Diccionario con las claves:
            - "scale": factor de escala aplicado en preprocesado.
            - "pad_left": padding horizontal aplicado.
            - "pad_top": padding vertical aplicado.
    Retorna:
        - Lista `p` con las coordenadas corregidas.
    """
    scale = transform_info["scale"]
    pad_left = transform_info["pad_left"]
    pad_top = transform_info["pad_top"]
    p[0] = (p[0] - pad_left) / scale
    p[1] = (p[1] - pad_top) / scale
    p[2] = (p[2] - pad_left) / scale
    p[3] = (p[3] - pad_top) / scale
    return p


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
al NMS a traves de el valor mutable "nms_threshold" que toma valor de "Reactive_output_config". Ya no se necesita
descongelar "postprocess" ya que no toma un valor estatico sino que lee el ultimo valor de confianza antes de 
hacer la funcion de confianza.
    Tengo que revisar el concepto del NMS, tengo un par de dudas respecto a como se aplica.
'''

'''
    Esta es la ultima parte del pipeline de salida, de aqui se tendrian que pasar los bouding boxes al cliente
el se debe encargar de dibujarlos sobre la imagen. No se toca la visualizacion, por lo que no necesito
devolver colores de los boxes ni la imagen del input pipeline en blanco y negro para devolverlo a valores normales. 
    Esa imagen ya puede desaparecer del flujo del programa.
'''