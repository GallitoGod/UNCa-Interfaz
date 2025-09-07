# test_output_transformer.py
"""
Tests del TRANSFORMADOR (postprocesador).
- Filtro de confianza (umbral en vivo).
- Top-K opcional.
- NMS (agnostico vs por clase).
- Undo del preprocesado (letterbox o resize directo).
"""

import numpy as np
import pytest

from api.func.output_pipeline.output_transformer import buildPostprocessor

class _DummyOutputTensor:
    def __init__(self, output_format: str):
        self.output_format = output_format

class _DummyOutputConfig:
    def __init__(
        self,
        output_format: str,
        *,
        confidence_threshold: float = 0.0,
        apply_confidence_filter=None,   # None -> default por formato
        apply_nms=None,                 # None -> default por formato
        nms_per_class: bool = False,
        iou_threshold: float = 0.5,
        top_k=None,
        pack_format= 'raw',
    ):
        self.output_tensor = _DummyOutputTensor(output_format)
        self.confidence_threshold = confidence_threshold
        self.apply_confidence_filter = apply_confidence_filter
        self.apply_nms = apply_nms
        self.nms_per_class = nms_per_class
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.pack_format = pack_format

class _DummyRuntime:
    def __init__(
        self,
        *,
        input_width=320, input_height=320,
        orig_width=320, orig_height=320,
        letterbox_used=False, scale=1.0, pad_left=0.0, pad_top=0.0,
    ):
        self.input_width = input_width
        self.input_height = input_height
        self.orig_width = orig_width
        self.orig_height = orig_height
        self.metadata_letter = {
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "letterbox_used": letterbox_used,
        }

def test_confidence_threshold_en_vivo():
    cfg = _DummyOutputConfig("yolo_flat", confidence_threshold=0.75,
                             apply_confidence_filter=True, apply_nms=False)
    rt = _DummyRuntime()
    post = buildPostprocessor(cfg, rt)

    rows = [
        [10, 10, 20, 20, 0.20, 0.0],
        [30, 30, 40, 40, 0.70, 1.0],
        [50, 50, 60, 60, 0.90, 2.0],
    ]
    out1 = post(rows)
    assert len(out1) == 1 and np.isclose(out1[0][4], 0.90)

    cfg.confidence_threshold = 0.5
    out2 = post(rows)
    assert len(out2) == 2
    assert np.allclose([r[4] for r in out2], [0.90, 0.70], atol=1e-6)

def test_nms_agnostico_vs_por_clase():
    # Agnostico
    cfg_a = _DummyOutputConfig("yolo_flat", confidence_threshold=0.0,
                               apply_confidence_filter=False, apply_nms=True,
                               nms_per_class=False, iou_threshold=0.5)
    # Por clase
    cfg_c = _DummyOutputConfig("yolo_flat", confidence_threshold=0.0,
                               apply_confidence_filter=False, apply_nms=True,
                               nms_per_class=True, iou_threshold=0.5)
    rt = _DummyRuntime()

    rows = [
        [0,   0,   100, 100, 0.9, 0.0],  # cls 0
        [10,  10,  110, 110, 0.8, 1.0],  # cls 1 (alto solape)
    ]

    post_a = buildPostprocessor(cfg_a, rt)
    post_c = buildPostprocessor(cfg_c, rt)

    out_a = post_a(rows)
    out_c = post_c(rows)

    assert len(out_a) == 1   # agnostico elimina una
    assert len(out_c) == 2   # por clase conserva ambas

def test_topk_prev_nms():
    cfg = _DummyOutputConfig("yolo_flat", confidence_threshold=0.0,
                             apply_confidence_filter=False, apply_nms=False, top_k=2)
    rt = _DummyRuntime()
    post = buildPostprocessor(cfg, rt)

    rows = [
        [0, 0, 10, 10, 0.1, 0.0],
        [2, 2, 12, 12, 0.9, 0.0],
        [4, 4, 14, 14, 0.8, 0.0],
        [6, 6, 16, 16, 0.7, 0.0],
    ]
    out = post(rows)
    assert len(out) == 2
    assert np.allclose([o[4] for o in out], [0.9, 0.8], atol=1e-6)

def test_undo_letterbox():
    cfg = _DummyOutputConfig("yolo_flat", apply_confidence_filter=False, apply_nms=False)
    rt = _DummyRuntime(
        input_width=320, input_height=320,
        orig_width=640, orig_height=360,
        letterbox_used=True, scale=0.5, pad_left=0.0, pad_top=70.0
    )
    post = buildPostprocessor(cfg, rt)

    rows = [[50, 95, 60, 105, 0.9, 1.0]]  # (x,y) tensor -> (100,50) original
    out = post(rows)
    x1, y1, x2, y2 = out[0][:4]
    assert np.isclose([x1, y1, x2, y2], [100, 50, 120, 70], atol=1e-6).all()

def test_undo_resize_directo():
    cfg = _DummyOutputConfig("yolo_flat", apply_confidence_filter=False, apply_nms=False)
    rt = _DummyRuntime(
        input_width=320, input_height=320,
        orig_width=640, orig_height=360,  # sx=2.0, sy=1.125
        letterbox_used=False
    )
    post = buildPostprocessor(cfg, rt)

    rows = [[160, 80, 200, 120, 0.8, 0.0]]
    out = post(rows)
    x1, y1, x2, y2 = out[0][:4]
    assert np.isclose([x1, y1, x2, y2], [320, 90, 400, 135], atol=1e-6).all()

def test_tflite_detpost_por_defecto_no_refiltra_ni_nms():
    cfg = _DummyOutputConfig("tflite_detpost", confidence_threshold=0.95,
                             apply_confidence_filter=None, apply_nms=None)
    rt = _DummyRuntime()
    post = buildPostprocessor(cfg, rt)

    rows = [
        [10, 10, 20, 20, 0.5, 0.0],
        [30, 30, 40, 40, 0.6, 1.0],
    ]
    out = post(rows)
    assert len(out) == 2  # se respetan tal cual
