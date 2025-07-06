from typing import Any, Callable, List, Union
import numpy as np

def build_unpacker(output_format: str) -> Callable[[Any], Union[List[List[float]], np.ndarray]]:
    """
    Devuelve una funcion especializada para desempaquetar el raw_output de un modelo
    segun el formato declarado en output_tensor.output_format.
    
    La funcion retornada convierte el output crudo en una estructura estandarizada
    para el adaptador de salida: List[List[float]] o np.ndarray.
    
    :param output_format: 'raw', 'multihead', o 'softmax'
    :return: Funcion desempaquetadora especializada
    """

    if output_format == "raw":
        return lambda r: r

    elif output_format == "multihead":
        return lambda r: np.concatenate([np.array(x) for x in r], axis=-1)

    elif output_format == "softmax":
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

    else:
        raise ValueError(f"output_format desconocido: {output_format}")