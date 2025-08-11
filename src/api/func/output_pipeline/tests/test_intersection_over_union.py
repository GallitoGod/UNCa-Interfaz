#from ...output_pipeline import iou
from api.func.output_pipeline import iou

def test_iou_no_overlap():
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]
    assert iou(box1, box2) == 0.0


def test_iou_partial_overlap():
    box1 = [0, 0, 10, 10]   # Area = 100
    box2 = [5, 5, 15, 15]   # Area = 100
    # Interseccion: cuadrado (5,5)-(10,10) → area = 25
    # Union = 100 + 100 - 25 = 175
    expected = 25 / 175
    assert iou(box1, box2) == expected


def test_iou_box_inside_another():
    box1 = [0, 0, 10, 10]   # Area = 100
    box2 = [2, 2, 8, 8]     # Area = 36
    expected = 36 / 100
    assert iou(box1, box2) == expected


def test_iou_identical_boxes():
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    assert iou(box1, box2) == 1.0


def test_iou_zero_area_boxes():
    box1 = [0, 0, 0, 0]
    box2 = [0, 0, 0, 0]
    assert iou(box1, box2) == 0
