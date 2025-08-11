#from ...output_pipeline import non_max_suppression
from api.func.output_pipeline import non_max_suppression

def test_nms_empty():
    assert non_max_suppression([], 0.5) == []
# Debe devolver lista vacia.


def test_nms_no_overlap():
    predictions = [
        [0, 0, 10, 10, 0.9],   # Caja A
        [20, 20, 30, 30, 0.8], # Caja B
    ]
    result = non_max_suppression(predictions, 0.5)
    assert len(result) == 2
    assert result[0][4] >= result[1][4]  # Sigue en orden de confianza
# Si no hay interseccion mayor al umbral, todas las cajas deben sobrevivir.


def test_nms_overlap_removes_lower_confidence():
    predictions = [
        [0, 0, 10, 10, 0.9],  # Caja A
        [1, 1, 9, 9, 0.8],    # Caja B (solapa mucho con A)
    ]
    result = non_max_suppression(predictions, 0.5)
    assert result == [[0, 0, 10, 10, 0.9]]
# Solo deberia quedar la de mayor confianza si el IoU supera el umbral.


def test_nms_overlap_below_threshold():
    predictions = [
        [0, 0, 10, 10, 0.9],
        [9, 9, 19, 19, 0.8],  # Solapa poco con la primera
    ]
    result = non_max_suppression(predictions, 0.5)
    assert len(result) == 2
# Ambas deben quedar si el IoU es menor al umbral.


def test_nms_orders_by_confidence():
    predictions = [
        [0, 0, 10, 10, 0.7],
        [0, 0, 10, 10, 0.95], # Misma caja que la anterior pero más confianza
        [20, 20, 30, 30, 0.5],
    ]
    result = non_max_suppression(predictions, 0.5)
    assert result[0][4] == 0.95
    assert all(result[i][4] >= result[i+1][4] for i in range(len(result)-1))
# Si entran desordenadas por confianza, deben salir ordenadas de mayor a menor.

