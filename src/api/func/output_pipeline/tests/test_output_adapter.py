# test_output_adapter.py
"""
Tests del ADAPTADOR (output_adapter).
Contrato probado:
- Dado un 'tensor_structure' (formato e indices), el adaptador debe convertir
  los 4 campos de caja del formato nativo a XYXY en el mismo orden [x1,y1,x2,y2].
- No toca score ni class_id (eso lo maneja el pipeline alrededor).
"""

import numpy as np
import pytest


from api.func.output_pipeline.output_adapter import _generate_box_converter


def test_convert_cxcywh_a_xyxy():
    """
    cx,cy,w,h -> xyxy
    cx=50, cy=40, w=20, h=10  =>  x1=40, y1=35, x2=60, y2=45
    """
    # Minima estructura necesaria
    box_format = "cxcywh"
    coordinates = {"cx": 0, "cy": 1, "w": 2, "h": 3}

    convert = _generate_box_converter(box_format, coordinates)

    row = [50.0, 40.0, 20.0, 10.0, 0.9, 3.0]  # [b0..b3, score, class]
    x1, y1, x2, y2 = convert(row)

    assert np.isclose([x1, y1, x2, y2], [40.0, 35.0, 60.0, 45.0], atol=1e-6).all()


def test_convert_yxyx_a_xyxy():
    """
    y1,x1,y2,x2 -> xyxy
    """
    box_format = "yxyx"
    coordinates = {"y1": 0, "x1": 1, "y2": 2, "x2": 3}

    convert = _generate_box_converter(box_format, coordinates)

    row = [10.0, 20.0, 30.0, 40.0, 0.5, 1.0]
    x1, y1, x2, y2 = convert(row)

    assert (x1, y1, x2, y2) == (20.0, 10.0, 40.0, 30.0)


def test_convert_xywh_a_xyxy():
    """
    xywh -> xyxy (x,y,w,h a esquina-superior-izq + tamaño)
    x=15, y=5, w=10, h=20  => x1=15, y1=5, x2=25, y2=25
    """
    box_format = "xywh"
    coordinates = {"x": 0, "y": 1, "w": 2, "h": 3}

    convert = _generate_box_converter(box_format, coordinates)

    row = [15.0, 5.0, 10.0, 20.0, 0.6, 2.0]
    x1, y1, x2, y2 = convert(row)

    assert (x1, y1, x2, y2) == (15.0, 5.0, 25.0, 25.0)
