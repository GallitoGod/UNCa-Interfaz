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
        def parse_efficientdet(raw_output, image_shape=None):
            """
            raw_output admite:
            - (boxes, class_scores[, ...])  ó
            - (class_scores, boxes[, ...])
            Donde:
            boxes:       (1, N, 4) en orden [ymin, xmin, ymax, xmax], normalizadas [0..1]
            class_scores:(1, N, C) probabilidades o logits ya softmaxeadas

            image_shape: (W, H) del tensor de entrada al modelo.

            Devuelve: List[List[float]] con [x1, y1, x2, y2, score, class_id] en floats.
            """
            a, b = raw_output[0], raw_output[1]

            # Detectar cual es boxes por el ultimo eje
            if a.shape[-1] == 4:
                boxes = a         # (1, N, 4)
                class_scores = b  # (1, N, C)
            else:
                class_scores = a
                boxes = b

            scores = class_scores[0]      # (N, C)
            bxy = boxes[0]                # (N, 4) [ymin,xmin,ymax,xmax]

            # Mejor clase por anchor (vectorizado) numpy
            best_cls = np.argmax(scores, axis=1)                            # (N,)
            best_sc = scores[np.arange(scores.shape[0]), best_cls]          # (N,)

            # Filtro por umbral (vectorizado) numpy
            mask = best_sc >= float(output.confidence_threshold)
            if not np.any(mask):
                return []

            sel = bxy[mask]                          # (K, 4)
            sc  = best_sc[mask].astype(np.float32)   # (K,)
            cl  = best_cls[mask].astype(np.float32)  # float por contrato del sistema

            ymin, xmin, ymax, xmax = sel.T
            x1, y1, x2, y2 = xmin, ymin, xmax, ymax  # a [x1,y1,x2,y2]

            # Escalado a pixeles del tensor de entrada
            if image_shape is not None:
                W, H = image_shape  
                x1 = x1 * W; x2 = x2 * W
                y1 = y1 * H; y2 = y2 * H

            dets = np.column_stack([x1, y1, x2, y2, sc, cl]).astype(float)

            return dets.tolist()
        return parse_efficientdet

    else:
        raise ValueError(f"output_format desconocido: {output.output_tensor.output_format}")