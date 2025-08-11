#from ...output_pipeline import undo_transform
from api.func.output_pipeline import undo_transform

def test_undo_transform_no_padding():
    p = [20, 30, 40, 50]
    transform_info = {"scale": 2, "pad_left": 0, "pad_top": 0}
    result = undo_transform(p.copy(), transform_info)
    assert result == [10, 15, 20, 25]  # Escalado simple sin padding


def test_undo_transform_with_padding():
    p = [25, 35, 45, 55]
    transform_info = {"scale": 2, "pad_left": 5, "pad_top": 5}
    result = undo_transform(p.copy(), transform_info)
    # Restar padding y luego escalar
    assert result == [10, 15, 20, 25]


def test_undo_transform_negative_padding():
    p = [15, 25, 35, 45]
    transform_info = {"scale": 2, "pad_left": -5, "pad_top": -5}
    result = undo_transform(p.copy(), transform_info)
    # Padding negativo significa que originalmente estaba mas recortado
    assert result == [10, 15, 20, 25]