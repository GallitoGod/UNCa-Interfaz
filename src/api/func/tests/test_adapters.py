
import numpy as np
from api.func.input_pipeline import generate_input_adapter
from api.func.reader_pipeline.config_schema import InputConfig, InputTensorConfig

def test_input_adapter():
    dummy_img = np.ones((640, 640, 3), dtype=np.uint8)
    cfg = InputConfig(
        width=640, height=640, channels=3, normalize=False, scale=False,
        input_tensor=InputTensorConfig(layout="NCHW", dtype="float32", quantized=False)
    )
    adapter = generate_input_adapter(cfg, runtime=None)
    out = adapter(dummy_img)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 3, 640, 640)
