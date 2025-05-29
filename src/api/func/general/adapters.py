from config_schema import TensorStructure
from coords_converter import generate_box_converter

def generate_output_adapter(tensor_structure: TensorStructure):
    convert_box = generate_box_converter(
        tensor_structure.format, tensor_structure.coordinates
    )

    conf_idx = tensor_structure.confidence_index
    cls_idx = tensor_structure.class_index
    
    def adapter_fn_out(row):
        box = convert_box(row)
        confidence = row[conf_idx]
        class_id = int(row[cls_idx])
        # En el orden que espera el postprocesador: [x1, y1, x2, y2, conf, class]
        return [*box, confidence, class_id]

    return adapter_fn_out




"""
    row viene de raw_output que viene directamente de la IA luego de la inferencia.
    Este contiene todas las detecciones que el modelo haya emitido, 
generalmente en forma de arrays, cada uno con informacion de una unica deteccion. El problema
esta cuando no viene en forma de List[List[float]] o np.ndarray, asi que el problema a resolver ahora es 
el que todas las IAs llegas este punto devuelvan estas dos clases de "listas" para que el adaptador funcione
correctamente.

    En el caso de ser List[List[float]] o np.ndarray, cada deteccion puede tener los datos en un orden diferente:
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
    Tambien hay que adaptarla si los modelos devuelven raw_outputs distintos de la forma
List[List[float]] o np.ndarray.    
"""