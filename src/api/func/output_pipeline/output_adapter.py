from typing import Callable
import numpy as np
from api.func.reader_pipeline.config_schema import TensorDetection


def _generate_box_converter(fmt: str, coords: dict) -> Callable[[np.ndarray], tuple]:
    """
    Devuelve una funcion (N,K) -> (x1, y1, x2, y2), cada uno una COLUMNA (vista 1D).

    Los indices se resuelven UNA vez, al armar el pipeline; la funcion devuelta solo
    hace slicing y aritmetica de numpy sobre el tensor entero.
    """
    if fmt in ("xyxy", "yxyx"):
        # yxyx comparte implementacion con xyxy y no es un descuido: el reordenamiento
        # ya lo hace el dict 'coordinates' del JSON (un modelo yxyx declara
        # {y1:0, x1:1, y2:2, x2:3}), asi que ambos formatos se leen igual — por nombre
        # de clave, no por posicion. Eran dos ramas identicas en la version fila-a-fila.
        i_x1, i_y1 = coords["x1"], coords["y1"]
        i_x2, i_y2 = coords["x2"], coords["y2"]

        def convertir(a):
            return a[:, i_x1], a[:, i_y1], a[:, i_x2], a[:, i_y2]

    elif fmt == "cxcywh":
        i_cx, i_cy, i_w, i_h = coords["cx"], coords["cy"], coords["w"], coords["h"]

        def convertir(a):
            cx, cy = a[:, i_cx], a[:, i_cy]
            # *0.5 y no /2 por gusto: en binario ambos son exactos (escalar por una
            # potencia de dos), asi que el resultado es bit a bit el mismo que antes.
            medio_w, medio_h = a[:, i_w] * 0.5, a[:, i_h] * 0.5
            return cx - medio_w, cy - medio_h, cx + medio_w, cy + medio_h

    elif fmt == "xywh":
        # No lo permite el Literal del schema (box_format solo acepta xyxy/cxcywh/yxyx),
        # asi que hoy es inalcanzable por config. Se conserva por paridad con la version
        # anterior: podar comportamiento es otra tarea, no la de vectorizar.
        i_x, i_y, i_w, i_h = coords["x"], coords["y"], coords["w"], coords["h"]

        def convertir(a):
            x, y = a[:, i_x], a[:, i_y]
            return x, y, x + a[:, i_w], y + a[:, i_h]

    else:
        raise ValueError(f"Formato desconocido: {fmt}")

    return convertir


def generate_output_adapter(tensor_structure: TensorDetection):
    """
    Reordena el tensor desempaquetado al formato estandar del sistema.

    **Contrato:** recibe la matriz ENTERA (N, K) y devuelve (N, 6) float32 con
    [x1, y1, x2, y2, conf, class_id]. NO es fila a fila.

    Por que trabaja sobre el tensor entero (2026-08-27): antes se llamaba una vez por
    fila desde un list comprehension del runner, o sea una llamada de Python por caja
    candidata. Con heads que ya traen NMS eso da igual —llegan un punado de filas—, pero
    con un head CRUDO llegan decenas de miles: `anchor_deltas` de efficientdet-lite0
    entrega ~19k anchors por frame y su postproceso costaba **37,7 ms, mas caro que su
    propia inferencia (21,4 ms)**. El trabajo real son cuatro restas y dos gathers de
    columnas: vectorizado es la misma cuenta sin el peaje del interprete.

    El JSON no cambia y la semantica tampoco: los indices siguen saliendo de
    'coordinates' / 'confidence_index' / 'class_index'.
    """
    fmt = tensor_structure.box_format or "xyxy"
    coords = tensor_structure.coordinates or {"x1": 0, "y1": 1, "x2": 2, "y2": 3}
    convertir_cajas = _generate_box_converter(fmt, coords)

    conf_idx = tensor_structure.confidence_index
    cls_idx = tensor_structure.class_index

    def _adapter_fn_out(unpacked) -> np.ndarray:
        a = np.asarray(unpacked, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)

        x1, y1, x2, y2 = convertir_cajas(a)

        # En el orden que espera el postprocesador: [x1, y1, x2, y2, conf, class]
        salida = np.empty((a.shape[0], 6), dtype=np.float32)
        salida[:, 0] = x1
        salida[:, 1] = y1
        salida[:, 2] = x2
        salida[:, 3] = y2
        salida[:, 4] = a[:, conf_idx]
        salida[:, 5] = a[:, cls_idx]
        return salida

    return _adapter_fn_out


"""
    'unpacked' viene de raw_output, que sale directamente de la IA luego de la
inferencia, y ya paso por el unpacker correspondiente (unpackers/): una fila por
deteccion candidata, en el orden de columnas que use ese modelo.

    Cada modelo puede traer los datos en un orden diferente:
[x1, y1, x2, y2, confidence, class_id] o
[cx, cy, w, h, conf, cls] o
[y1, x1, y2, x2, ...], etc.

    El adaptador lee en el orden correcto segun el JSON, reestructura si hace falta, y
devuelve el formato estandar que el sistema entiende: [x1, y1, x2, y2, conf, class_id].
Ese resultado lo consume el postprocesador (conf filter -> top-k -> NMS -> undo del
letterbox) antes de convertirse en sv.Detections.
"""
