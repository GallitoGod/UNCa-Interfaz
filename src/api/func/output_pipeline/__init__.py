from .output_adapter import generate_output_adapter
from .output_transformer import buildPostprocessor, build_classification_postprocessor

__all__ = [
    'generate_output_adapter',
    'buildPostprocessor',
    'build_classification_postprocessor',
]

'''
    Output_pipeline convierte la salida cruda del modelo en el resultado de dominio
que el cliente puede consumir. Hay un postprocesador por familia de modelo:

    - buildPostprocessor                  -> DETECCION. Devuelve cajas
      [x1, y1, x2, y2, conf, cls] en pixeles de la imagen original.
    - build_classification_postprocessor  -> CLASIFICACION. Devuelve (K, 2)
      [class_id, score] ordenado por score desc.

    El DIBUJO no es responsabilidad del postproceso sino de quien consume el output:
        Una app cliente (web, movil, escritorio)
        Un visor de detecciones (por ejemplo, un servicio de monitoreo en tiempo real)
        Una funcion de visualizacion (draw_boxes(img, detections))

    Ellos toman las coordenadas [x1, y1, x2, y2], y dibujan las cajas sobre la imagen
original capturada por la camara.

    NOTA HISTORICA: hasta 2026-08-13 todo este paquete estaba estandarizado SOLO en
deteccion de objetos. La clasificacion ya tiene su unpacker (unpackers/classification.py)
y su postprocesador. Falta segmentacion (decode de mascara).
'''
