from config_schema import ModelConfig

def generate_output_adapter(config: ModelConfig):
    structure = config.output.tensor_structure
    conf_thres = config.output.confidence_threshold

    def output_adapter(raw_output):
        detections = []

        for d in raw_output:
            box = [d[i] for i in structure.box_indices]
            confidence = d[structure.confidence_index]
            class_id = int(d[structure.class_index])

            # En el orden que espera el postprocesador: [x1, y1, x2, y2, conf, class]
            detections.append(box + [confidence, class_id])

        return detections

    return output_adapter




"""
    raw_output viene directamente de la IA luego de la inferencia.
Este contiene todas las detecciones que el modelo haya emitido, 
generalmente en forma de arrays, cada uno con informacion de una unica deteccion.

    Cada deteccion puede tener los datos en un orden diferente:
[x1, y1, x2, y2, confidence, class_id] o
[cx, cy, w, h, conf, cls] o
[y1, x1, y2, x2, ...], etc.

    El adaptador se encarga de leer en el orden correcto, 
reestructurar si hace falta, y devolver en el formato estandar que el sistema entiende:
[x1, y1, x2, y2, conf, class_id].
    Este resultado lo va a obtener el controlador y se lo va a pasar al postprocesador. Luego, lo unico que
faltaria seria devolverlo al cliente.
"""

"""
    La idea seria hacer algo asi, solo que se mantenga congelado en una unica opcion
hasta que se cambie de IA. Como ya pasa con los pasos a seguir los postprocesos y preprocesos genericos
del controlador. Basicamente, no solo que es posible permitir una mayor flexibilidad a la hora de 
utilizar coordenadas distintas en distintas IAs, sino que es necesario para mantener la filosofia del programa;
la capacidad de poder hacer andar la mayor cantidad de IAs CASI independientes de su formato, estructura y variables.
"""
def convert_box(box, format: str, coordinates: dict):
    if format == "xyxy":
        x1 = box[coordinates["x1"]]
        y1 = box[coordinates["y1"]]
        x2 = box[coordinates["x2"]]
        y2 = box[coordinates["y2"]]
    elif format == "cxcywh":
        cx = box[coordinates["cx"]]
        cy = box[coordinates["cy"]]
        w = box[coordinates["w"]]
        h = box[coordinates["h"]]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
    elif format == "yxyx":
        y1 = box[coordinates["y1"]]
        x1 = box[coordinates["x1"]]
        y2 = box[coordinates["y2"]]
        x2 = box[coordinates["x2"]]
    else:
        raise ValueError(f"Unknown box format: {format}")
    
    return [x1, y1, x2, y2]
#   Si esta, hasta ahora, funcion resulta ser muy grande, incluso se podria separar en otro script.

"""
    Hay que alterar la lectura de los JSONs a:
    "tensor_structure": {
        "format": "yxyx",       |   "format": "cxcywh",
        "coordinates": {        |   "coordinates": {
            "y1": 0,            |       "cx": 0,
            "x1": 1,            |       "cy": 1,
            "y2": 2,            |       "w": 2,
            "x2": 3             |       "h": 3
        },                      |   },
        "confidence_index": 4,
        "class_index": 5
    }
    Y con ello ya podria implementarse esta idea.
"""