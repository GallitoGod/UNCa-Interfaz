import os
import json
from config_schema import ModelConfig

def loadModelConfig(model_path: str):
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    config_path = os.path.join("configs", base_name + ".json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontro config para {base_name}")
    with open(config_path, "r") as f:
        raw = json.load(f)
    return ModelConfig(**raw)

