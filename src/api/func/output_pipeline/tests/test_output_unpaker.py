# tests/test_output_unpacker.py
import numpy as np
import pytest

from api.func.output_pipeline.unpackers.registry import unpack_out
from api.func.reader_pipeline.config_schema import RuntimeConfig, RuntimeShapes

class _DummyOutputConfig:
    def __init__(self, pack_format: str):
        self.pack_format = pack_format


def _rt(*, iw=320, ih=320, ow=320, oh=320, out_coords_space="tensor_pixels"):
    return RuntimeConfig(
        runtimeShapes=RuntimeShapes(
            input_width=iw,
            input_height=ih,
            orig_width=ow,
            orig_height=oh,
            out_coords_space=out_coords_space,
        )
    )


def test_yolo_flat_tensor_pixels_escalado():
    cfg = _DummyOutputConfig("yolo_flat")
    rt = _rt(out_coords_space="tensor_pixels", iw=320, ih=320)

    # raw NORMALIZADO (cx,cy,w,h) -> debe escalar a pixeles del tensor
    raw = np.array([[0.5, 0.5, 0.2, 0.1, 0.9, 0.1, 0.8, 0.2]], dtype=np.float32)  # score=0.72, cls=1

    fn = unpack_out(cfg)
    arr = fn(raw, rt)
    assert isinstance(arr, np.ndarray) and arr.shape == (1, 6)

    cx, cy, w, h, sc, cl = arr[0].tolist()
    assert np.allclose([cx, cy, w, h], [160, 160, 64, 32], atol=1e-5)
    assert np.isclose(sc, 0.72) and cl == 1.0


def test_yolo_flat_normalized_no_escalado():
    cfg = _DummyOutputConfig("yolo_flat")
    rt = _rt(out_coords_space="normalized_0_1", iw=320, ih=320)

    raw = np.array([[0.5, 0.5, 0.2, 0.1, 0.9, 0.1, 0.8, 0.2]], dtype=np.float32)

    fn = unpack_out(cfg)
    arr = fn(raw, rt)
    cx, cy, w, h, sc, cl = arr[0].tolist()
    assert np.allclose([cx, cy, w, h], [0.5, 0.5, 0.2, 0.1], atol=1e-6)
    assert np.isclose(sc, 0.72) and cl == 1.0


def test_boxes_scores_detecta_orden_y_escalado_tensor_pixels():
    cfg = _DummyOutputConfig("boxes_scores")
    rt = _rt(out_coords_space="tensor_pixels", iw=320, ih=320)

    boxes = np.array([
        [0.1, 0.2, 0.5, 0.6],   # y1,x1,y2,x2 normalizado
        [0.0, 0.0, 1.0, 1.0],
    ], dtype=np.float32)
    scores = np.array([
        [0.1, 0.7, 0.2],        # best=1 (0.7)
        [0.9, 0.05, 0.05],      # best=0 (0.9)
    ], dtype=np.float32)

    fn = unpack_out(cfg)

    # (boxes, scores)
    arr1 = fn((boxes, scores), rt)
    # (scores, boxes)
    arr2 = fn((scores, boxes), rt)

    for arr in (arr1, arr2):
        assert arr.shape == (2, 6)

        y1, x1, y2, x2, sc, cl = arr[0].tolist()
        assert np.allclose([y1, x1, y2, x2], [32, 64, 160, 192], atol=1e-5)
        assert np.isclose(sc, 0.7) and cl == 1.0

        y1b, x1b, y2b, x2b, scb, clb = arr[1].tolist()
        assert np.allclose([y1b, x1b, y2b, x2b], [0, 0, 320, 320], atol=1e-5)
        assert np.isclose(scb, 0.9) and clb == 0.0


def test_tflite_detpost_respeta_count_y_escalado_tensor_pixels():
    cfg = _DummyOutputConfig("tflite_detpost")
    rt = _rt(out_coords_space="tensor_pixels", iw=320, ih=320)

    boxes = np.array([
        [0.2, 0.2, 0.4, 0.5],  # ymin,xmin,ymax,xmax normalizado
        [0.0, 0.0, 1.0, 1.0],
    ], dtype=np.float32)
    scores = np.array([0.8, 0.9], dtype=np.float32)
    classes = np.array([3.0, 1.0], dtype=np.float32)
    count = np.array([1], dtype=np.int32)

    fn = unpack_out(cfg)
    arr = fn((boxes, scores, classes, count), rt)

    assert arr.shape == (1, 6)
    y1, x1, y2, x2, sc, cl = arr[0].tolist()
    assert np.allclose([y1, x1, y2, x2], [64, 64, 128, 160], atol=1e-5)
    assert np.isclose(sc, 0.8) and cl == 3.0


# def test_anchor_deltas_decodifica_y_devuelve_pixeles_tensor():
#     cfg = _DummyOutputConfig("anchor_deltas")
#     rt = _rt(out_coords_space="tensor_pixels", iw=320, ih=320)

#     # Nota: anchor_deltas usa runtime.anchors y runtime.box_variance (hoy no estan en RuntimeShapes)
#     # Esto esta bien como "runtime extendido" (atributos extra).
#     rt.anchors = np.array([
#         [0.5,  0.5,  0.2, 0.1],
#         [0.25, 0.25, 0.1, 0.2],
#     ], dtype=np.float32)
#     rt.box_variance = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)

#     deltas_2d = np.zeros((2, 4), dtype=np.float32)
#     logits_2d = np.array([
#         [0.0, 10.0, 0.0],   # cls 1
#         [10.0, 0.0, 0.0],   # cls 0
#     ], dtype=np.float32)

#     fn = unpack_out(cfg)
#     arr = fn((deltas_2d, logits_2d), rt)
#     assert arr.shape == (2, 6)

#     # arr es [ymin,xmin,ymax,xmax,score,class]
#     exp = np.array([
#         [128.0, 144.0, 192.0, 176.0],
#         [ 64.0,  48.0,  96.0, 112.0],
#     ], dtype=np.float32)

#     np.testing.assert_allclose(arr[:, 0:4], exp, atol=1e-4)
#     assert arr[0, 5] == 1.0 and arr[0, 4] > 0.99
#     assert arr[1, 5] == 0.0 and arr[1, 4] > 0.99