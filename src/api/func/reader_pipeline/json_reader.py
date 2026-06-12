import os
import json
from pathlib import Path
from typing import Optional, Union
from .config_schema import ModelConfig

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "configs"

def load_model_config(model_path: str, configs_dir: Optional[Union[str, Path]] = None) -> ModelConfig:
    """
    Carga configs/<basename>.json y lo valida contra el schema.
    configs_dir permite inyectar otro directorio (tests); por defecto usa configs/ del proyecto.
    """
    cfg_dir = Path(configs_dir) if configs_dir is not None else _CONFIGS_DIR
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    config_path = cfg_dir / (base_name + ".json")
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontro configuracion para '{base_name}' en {cfg_dir}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return ModelConfig(**raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error al decodificar el JSON de configuracion: {e}") from e

