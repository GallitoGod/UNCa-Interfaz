#from ...output_pipeline import generate_box_converter
from api.func.output_pipeline import generate_box_converter
import pytest

def test_generate_box_converter_xyxy():
    row = [10, 20, 30, 40]  # x1, y1, x2, y2
    coords = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}
    convert = generate_box_converter("xyxy", coords)
    assert convert(row) == [10, 20, 30, 40]


def test_generate_box_converter_cxcywh():
    row = [15, 25, 10, 20]
    coords = {"cx": 0, "cy": 1, "w": 2, "h": 3}
    convert = generate_box_converter("cxcywh", coords)
    # cx - w/2, cy - h/2, cx + w/2, cy + h/2
    assert convert(row) == [10, 15, 20, 35]


def test_generate_box_converter_yxyx():
    row = [20, 10, 40, 30]
    coords = {"y1": 0, "x1": 1, "y2": 2, "x2": 3}
    converter = generate_box_converter("yxyx", coords)
    assert converter(row) == [10, 20, 30, 40]


def test_generate_box_converter_invalid_format():
    coords = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}
    with pytest.raises(ValueError, match="Formato desconocido: invalid"):
        generate_box_converter("invalid", coords)

'''
Si ya el cliente y las APIs garantizan que la entrada viene limpia, se asume que:
    Las coordenadas existen y estan en el formato esperado.
    La confianza y la clase tienen valores validos.
    No van a llegar datos corruptos o fuera de rango.
Por lo que no hace falta escuchar errores en rangos de confianzas o clases corruptas.
'''