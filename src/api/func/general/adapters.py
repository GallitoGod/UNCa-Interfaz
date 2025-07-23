from api.func.general.tensor_converter import generate_box_converter, generate_layout_converter
from general.config_schema import TensorStructure, InputConfig
import numpy as np

def generate_input_adapter(input_config: InputConfig):
    tensor_cfg = input_config.input_tensor or None
    color_order = input_config.color_order or "RGB"
    layout_converter = generate_layout_converter(tensor_cfg.layout) if tensor_cfg else lambda x: x
    dtype = tensor_cfg.dtype if tensor_cfg else "float32"

    if color_order == "BGR":
        def adapter_fn_in(img: np.ndarray) -> np.ndarray:
            img = img[..., ::-1]           # invierte canales
            img = layout_converter(img)    # reordenar layout
            return img.astype(dtype)       # tipo final
    else:
        def adapter_fn_in(img: np.ndarray) -> np.ndarray:
            img = layout_converter(img)
            return img.astype(dtype)

    return adapter_fn_in



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
generalmente en forma de arrays, cada uno con informacion de una unica deteccion. raw_output 
puede no venir en forma de List[List[float]] o np.ndarray, asi que el problema a resolver ahora es 
el que todas las IAs llegado este punto devuelvan estas dos clases de "listas" para que el adaptador funcione
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
List[List[float]] o np.ndarray. <---    ESTO YA ESTA HECHO CON 'unpackers.py'   
"""