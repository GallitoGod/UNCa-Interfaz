import os
import json
from pathlib import Path
from .config_schema import ModelConfig

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "configs"

def load_model_config(model_path: str) -> ModelConfig:
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    config_path = _CONFIGS_DIR / (base_name + ".json")
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontro configuracion para '{base_name}' en {_CONFIGS_DIR}")
    try:
        with open(config_path, "r") as f:
            raw = json.load(f)
        return ModelConfig(**raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error al decodificar el JSON de configuracion: {e}") from e

