from .output_adapter import generate_output_adapter
from .output_transformer import buildPostprocessor
from .output_unpacker import unpack_out

__all__ = [
    'generate_output_adapter', 
    'buildPostprocessor', 
    'unpack_out'
]

'''
    Output_pipeline devuelve los bouding boxes generados por la IA de la manera correcta para poder ser 
aplicados sobre la imagen original por el cliente.

    La superposicion de las cajas no es responsabilidad del postproceso, sino de quien consume el output, por ejemplo:
        Una app cliente (web, movil, escritorio)
        Un visor de detecciones (por ejemplo, un servicio de monitoreo en tiempo real)
        Una funcion de visualizacion (draw_boxes(img, detections))

    Ellos toman las coordenadas [x1, y1, x2, y2], y dibujan las cajas sobre la imagen original capturada por la camara.

    ES IMPORTANTE RECALCAR QUE TANTO INPUT_PIPELINE COMO OUTPUT_PIPELINE ESTAN ESTANDARIZADOS EN DETECCION DE OBJETOS.
    UNA VEZ TODA ESTA RUTA DE ENSAMBLAJE ESTE COMPLETADA, SE VA A EMPEZAR A ACTUALIZAR LA APLICACION PARA TENER PIPELINES
DE CLASIFICACION, SEGMENTACION, ETC.
'''