#from ...output_pipeline import generate_output_adapter
from api.func.output_pipeline import generate_output_adapter
#from ...reader_pipeline import TensorStructure
#from api.func.reader_pipeline.config_schema import TensorStructure
import pytest

class DummyTensorStructure:
    def __init__(self, box_format=None, coordinates=None, confidence_index=None, class_index=None, num_classes=None):
        self.box_format = box_format
        self.coordinates = coordinates
        self.confidence_index = confidence_index
        self.class_index = class_index
        self.num_classes = num_classes

def test_generate_output_adapter_FCCoCl(): # Orden de entrada: formato, coordenadas, confianza, clases 
    row = [10, 20, 30, 40, 0.5, 1]
    coords = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}
    confidence_index = 4
    class_index = 5
    num_classes = None
    tensor_structure = DummyTensorStructure("xyxy", coords, confidence_index, class_index, num_classes)
    adapter = generate_output_adapter(tensor_structure)
    adapted = adapter(row)
    assert adapted == [10, 20, 30, 40, 0.5, 1] # Orden esperado: [x1, y1, x2, y2, conf, class]


def test_generate_output_adapter_FCClCo(): # Orden de entrada: formato, coordenadas, clases, confianza 
    row = [10, 20, 30, 40, 1, 0.5]
    coords = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}
    confidence_index = 5
    class_index = 4
    num_classes = None
    tensor_structure = DummyTensorStructure("xyxy", coords, confidence_index, class_index, num_classes)
    adapter = generate_output_adapter(tensor_structure)
    adapted = adapter(row)
    assert adapted == [10, 20, 30, 40, 0.5, 1] # Orden esperado: [x1, y1, x2, y2, conf, class]

def test_generate_output_adapter_invalid_format():
    coords = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}
    confidence_index = 4
    class_index = 5
    num_classes = None
    tensor_structure = DummyTensorStructure("invalid", coords, confidence_index, class_index, num_classes)
    with pytest.raises(ValueError, match="Formato desconocido: invalid"):
        generate_output_adapter(tensor_structure)