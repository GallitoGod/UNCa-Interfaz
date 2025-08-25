from api.func.reader_pipeline.config_schema import OutputConfig
from typing import Any, Callable, List, Union
import numpy as np

def unpack_out(output: OutputConfig) -> Callable[[Any], Union[List[List[float]], np.ndarray]]:
    """
    Devuelve una funcion especializada para desempaquetar el raw_output de un modelo
    segun el formato declarado en output_tensor.output_format.
    
    La funcion retornada convierte el output crudo en una estructura estandarizada
    para el adaptador de salida: List[List[float]] o np.ndarray.
    
    :param output_format: 'raw', 'multihead', 'softmax' o 'efficientdet'.
    :return: Funcion desempaquetadora especializada
    """

    if output.output_tensor.output_format == "raw":
        return lambda r: r

    elif output.output_tensor.output_format == "multihead":     
        #   <- Cada feature map produce una deteccion independiente, por ello se terminan concatenando
        return lambda r: np.concatenate([np.array(x) for x in r], axis=-1)

    elif output.output_tensor.output_format == "softmax":       
        #   <- No es muy relevante en deteccion, terminara en clasificacion
        def unpack_softmax(raw_output):
            arr = np.array(raw_output)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            results = []
            for row in arr:
                class_id = int(np.argmax(row))
                confidence = float(np.max(row))
                results.append([0.0, 0.0, 0.0, 0.0, confidence, class_id])
            return results
        return unpack_softmax
    
    elif output.output_tensor.output_format == "efficientdet":
        def parse_efficientdet(raw_output, image_shape=(1920, 1080)):
            """
            Convierte la salida de un modelo EfficientDet en detecciones listas.
            Devuelve lista de listas en el formato:
                [x_min, y_min, x_max, y_max, score, class_id]
            Esta funcion filtra confianza antes del transformador para un mejor
                 manejo y menor consumo de potencia de procesador.
            """
            boxes, scores, classes, count = raw_output
            detections = []
            h, w = image_shape

            for i in range(boxes.shape[1]): 
                scores = classes[0, i]   # Probabilidades para todas las clases
                class_id = np.argmax(scores)    # Clase con mayor score
                score = scores[class_id]

                if score < output.confidence_threshold:
                    continue  # Descarta detecciones de baja confianza

                # Coordenadas normalizadas
                y_min, x_min, y_max, x_max = boxes[0, i]

                # Escala los boxes (1920x1080)
                x_min = int(x_min * w)
                x_max = int(x_max * w)
                y_min = int(y_min * h)
                y_max = int(y_max * h)

                # Guarda como List[List[float]]
                detections.append([x_min, y_min, x_max, y_max, float(score), int(class_id)])
            return detections
        return parse_efficientdet

    else:
        raise ValueError(f"output_format desconocido: {output.output_tensor.output_format}")