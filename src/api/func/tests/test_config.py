
from api.func.general.config_schema import ModelConfig
import json

def test_load_config():
    with open("configs/yolov7-tiny.json") as f:
        raw = json.load(f)
    cfg = ModelConfig(**raw)
    assert cfg.input.width > 0
    assert cfg.output.tensor_structure is not None
