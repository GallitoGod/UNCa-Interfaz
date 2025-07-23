
from api.func.model_controller import ModelController
import numpy as np

def test_full_inference():
    mc = ModelController()
    mc.load_model("models/yolov7-tiny.onnx")
    dummy_img = np.ones((640, 640, 3), dtype=np.uint8)
    results = mc.inference(dummy_img)
    assert isinstance(results, list)
