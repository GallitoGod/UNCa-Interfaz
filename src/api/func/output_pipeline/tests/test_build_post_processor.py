#from ...output_pipeline import buildPostprocessor
from api.func.output_pipeline import buildPostprocessor


class DummyTransformInfo:
    def __init__(self, letterbox_used=False, scale=1, pad_left=0, pad_top=0):
        self.metadata_letter = {
            "letterbox_used": letterbox_used,
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
        }


class DummyConfig:
    def __init__(self, confidence_threshold=0.5, apply_nms=False, nms_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.apply_nms = apply_nms
        self.nms_threshold = nms_threshold


def test_postprocessor_confidence_only():
    config = DummyConfig(confidence_threshold=0.6, apply_nms=False)
    transform_info = DummyTransformInfo(letterbox_used=False)
    post = buildPostprocessor(config, transform_info)

    preds = [
        [0, 0, 10, 10, 0.5],  # Debe filtrarse
        [0, 0, 10, 10, 0.7],  # Debe quedarse
    ]
    result = post(preds)
    assert result == [[0, 0, 10, 10, 0.7]]


def test_postprocessor_with_nms():
    config = DummyConfig(confidence_threshold=0.0, apply_nms=True, nms_threshold=0.5)
    transform_info = DummyTransformInfo(letterbox_used=False)
    post = buildPostprocessor(config, transform_info)

    preds = [
        [0, 0, 10, 10, 0.9],
        [1, 1, 9, 9, 0.8],  # Alta superposicion → eliminado por NMS
    ]
    result = post(preds)
    assert len(result) == 1
    assert result[0][4] == 0.9


def test_postprocessor_with_letterbox():
    config = DummyConfig(confidence_threshold=0.0, apply_nms=False)
    transform_info = DummyTransformInfo(letterbox_used=True, scale=2, pad_left=0, pad_top=0)
    post = buildPostprocessor(config, transform_info)

    preds = [
        [20, 30, 40, 50, 0.9],
    ]
    result = post(preds)
    assert result == [[10, 15, 20, 25, 0.9]]
