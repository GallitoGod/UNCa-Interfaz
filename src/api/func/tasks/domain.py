# tasks/domain.py — el TIPO DE DOMINIO de las tareas geometricas (deteccion, y
# manana segmentacion): sv.Detections de supervision.
#
# Por que existe este archivo (paso 2 del plan del 2026-08-21):
# el pipeline interno seguia moviendo un ndarray (N,6) pelado. Ese contrato no
# tiene donde poner mascara, tracker_id ni metadatos, y sobre todo NO es lo que
# consumen los annotators/ByteTrack/PolygonZone de supervision, que es a donde
# vamos en el paso 3 (render en el backend). Aca se concentra la conversion en
# UN solo lugar, detras del seam de tasks/, para que el resto del pipeline
# (unpackers, adapter, postproceso) siga hablando ndarray y no se entere.
#
# Lo que supervision NO hace, y por eso el resto del pipeline se queda:
# sv.Detections es un CONTENEDOR, no un interprete de tensores. Exige
# xyxy/confidence/class_id ya calculados; sus from_* cubren librerias concretas
# (ultralytics, transformers...), nunca un tensor crudo descrito por un JSON.
# Eso es justamente lo que hace UNCaLens.

import numpy as np
import supervision as sv


def empty_detections() -> sv.Detections:
    """
    sv.Detections vacio pero con los tres campos POBLADOS (arrays de largo 0).

    No se usa sv.Detections.empty() a proposito: ese deja confidence/class_id en
    None y obliga a chequear None en cada consumidor. Con arrays vacios el
    codigo de abajo (serializar, anotar, contar) es el mismo con 0 o con N cajas.
    """
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )


def detections_from_array(arr) -> sv.Detections:
    """
    (N,K>=6) [x1,y1,x2,y2,conf,cls] en px de la imagen original -> sv.Detections.

    Es la frontera de entrada al tipo de dominio: la ultima linea del pipeline de
    deteccion. Columnas extra (K>6) se descartan; el contrato documentado siempre
    fue (N,6) y nadie rio abajo las leia.

    Levanta ValueError si la forma no es la declarada: preferimos el error visible
    al frame mudo (misma politica que el schema estricto).
    """
    a = np.asarray(arr, dtype=np.float32)

    # Sin cajas: caso normal (frame sin nada por encima del umbral), no un error.
    if a.size == 0:
        return empty_detections()

    if a.ndim != 2 or a.shape[1] < 6:
        raise ValueError(
            "detections_from_array espera una matriz (N,6) [x1,y1,x2,y2,conf,cls]; "
            f"recibio shape={a.shape}."
        )

    return sv.Detections(
        # ascontiguousarray: los slices de columnas son vistas no contiguas y los
        # annotators de supervision escriben/leen por buffer.
        xyxy=np.ascontiguousarray(a[:, :4], dtype=np.float32),
        confidence=np.ascontiguousarray(a[:, 4], dtype=np.float32),
        # class_id es entero por contrato de supervision (indexa paletas y label_map).
        class_id=a[:, 5].astype(int),
    )


def array_from_detections(dets: sv.Detections) -> np.ndarray:
    """
    sv.Detections -> ndarray (N,6) [x1,y1,x2,y2,conf,cls].

    Frontera de SALIDA, para lo que todavia habla en filas: el serializador del
    envelope y los tests. Mientras el cliente siga dibujando (hasta el paso 3),
    este es el camino por el que las cajas llegan al WebSocket.
    """
    if not isinstance(dets, sv.Detections):
        raise TypeError(
            f"array_from_detections espera sv.Detections, recibio {type(dets).__name__}."
        )

    n = len(dets)
    if n == 0:
        return np.empty((0, 6), dtype=np.float32)

    # confidence/class_id pueden venir en None si el Detections lo armo otro
    # (p.ej. sv.Detections.empty() o un from_* de supervision): se completan con
    # valores neutros en vez de reventar.
    conf = dets.confidence
    conf = np.zeros(n, dtype=np.float32) if conf is None else np.asarray(conf, dtype=np.float32)
    cls = dets.class_id
    cls = np.zeros(n, dtype=np.float32) if cls is None else np.asarray(cls, dtype=np.float32)

    out = np.empty((n, 6), dtype=np.float32)
    out[:, :4] = dets.xyxy
    out[:, 4] = conf
    out[:, 5] = cls
    return out
