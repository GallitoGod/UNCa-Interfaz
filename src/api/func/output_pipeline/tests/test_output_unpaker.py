# test_output_unpacker.py
"""
Tests del DESEMPAQUEDOR (unpacker).
- Sin filtrado por confianza.
- Escalado guiado por runtime.out_coords_space:
    "normalized_0_1" -> escala a pixeles del tensor
    "tensor_pixels"  -> deja tal cual
- Formatos: yolo_flat, boxes_scores, tflite_detpost
"""

import numpy as np
import pytest

from api.func.output_pipeline.output_unpacker import unpack_out

class _DummyOutputConfig:
    def __init__(self, pack_format: str):
        self.pack_format = pack_format
class _DummyRuntime:
    def __init__(
        self,
        *,
        input_width=320, input_height=320,
        orig_width=320, orig_height=320,
        out_coords_space="tensor_pixels",  # "normalized_0_1" | "tensor_pixels"
        letterbox_used=False, scale=1.0, pad_left=0.0, pad_top=0.0,
    ):
        self.input_width = input_width
        self.input_height = input_height
        self.orig_width = orig_width
        self.orig_height = orig_height
        self.out_coords_space = out_coords_space
        self.metadata_letter = {
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "letterbox_used": letterbox_used,
        }

def test_yolo_flat_tensor_pixels_no_escalado():
    cfg = _DummyOutputConfig("yolo_flat")
    rt = _DummyRuntime(out_coords_space="tensor_pixels", input_width=320, input_height=320)

    # (N,5+C) -> [cx, cy, w, h, obj, cls...]
    raw = np.array([
        [160, 160, 64, 32, 0.9, 0.1, 0.8, 0.2],  # best cls=1, score=0.9*0.8=0.72
    ], dtype=np.float32)

    fn = unpack_out(cfg)
    rows = fn(raw, rt)
    cx, cy, w, h, sc, cl = rows[0]
    assert (cx, cy, w, h) == (160, 160, 64, 32)
    assert np.isclose(sc, 0.72) and cl == 1.0

def test_yolo_flat_normalized_con_escalado():
    cfg = _DummyOutputConfig("yolo_flat")
    rt = _DummyRuntime(out_coords_space="normalized_0_1", input_width=320, input_height=320)

    raw = np.array([
        [0.5, 0.5, 0.2, 0.1, 0.9, 0.1, 0.8, 0.2],  # -> cx=160, cy=160, w=64, h=32
    ], dtype=np.float32)

    fn = unpack_out(cfg)
    rows = fn(raw, rt)
    cx, cy, w, h, sc, cl = rows[0]
    assert np.isclose([cx, cy, w, h], [160, 160, 64, 32]).all()
    assert np.isclose(sc, 0.72) and cl == 1.0

def test_boxes_scores_detecta_orden_y_normaliza():
    cfg = _DummyOutputConfig("boxes_scores")
    print(cfg)
    rt = _DummyRuntime(out_coords_space="normalized_0_1", input_width=320, input_height=320)

    boxes = np.array([
        [0.1, 0.2, 0.5, 0.6],   # [y1=32, x1=64, y2=160, x2=192]
        [0.0, 0.0, 1.0, 1.0],
    ], dtype=np.float32)
    scores = np.array([
        [0.1, 0.7, 0.2],        # best=1 (0.7)
        [0.9, 0.05, 0.05],      # best=0 (0.9)
    ], dtype=np.float32)

    fn = unpack_out(cfg)

    # orden (boxes, scores)
    rows1 = fn((boxes, scores), rt)
    # orden (scores, boxes)
    rows2 = fn((scores, boxes), rt)

    for rows in (rows1, rows2):
        y1, x1, y2, x2, sc, cl = rows[0]
        assert np.isclose([y1, x1, y2, x2], [32, 64, 160, 192]).all()
        assert np.isclose(sc, 0.7) and cl == 1.0

        y1b, x1b, y2b, x2b, scb, clb = rows[1]
        assert np.isclose([y1b, x1b, y2b, x2b], [0, 0, 320, 320]).all()
        assert np.isclose(scb, 0.9) and clb == 0.0

def test_tflite_detpost_respeta_count_y_normaliza():
    cfg = _DummyOutputConfig("tflite_detpost")
    print(cfg.pack_format)
    rt = _DummyRuntime(out_coords_space="normalized_0_1", input_width=320, input_height=320)

    boxes = np.array([
        [0.2, 0.2, 0.4, 0.5],  # -> [64,64,128,160]
        [0.0, 0.0, 1.0, 1.0],
    ], dtype=np.float32)
    scores = np.array([0.8, 0.9], dtype=np.float32)
    classes = np.array([3.0, 1.0], dtype=np.float32)
    count = np.array([1], dtype=np.int32)

    fn = unpack_out(cfg)
    rows = fn((boxes, scores, classes, count), rt)

    assert len(rows) == 1
    y1, x1, y2, x2, sc, cl = rows[0]
    assert np.isclose([y1, x1, y2, x2], [64, 64, 128, 160]).all()
    assert np.isclose(sc, 0.8) and cl == 3.0
