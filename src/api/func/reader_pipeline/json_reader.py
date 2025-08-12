import os
import json
from .config_schema import ModelConfig

def load_model_config(model_path: str) -> ModelConfig:
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    config_path = os.path.join("configs", base_name + ".json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontro configuracion para {base_name}")
    try:
        with open(config_path, "r") as f:
            raw = json.load(f)
        return ModelConfig(**raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error al decodificar el JSON de configuracion: {e}") from e

