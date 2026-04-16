# logger.py
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from typing import Optional, Dict, Any
import numpy as np
import os

# Rotacion: cada archivo crece hasta MAX_BYTES; se conservan BACKUP_COUNT
# archivos anteriores. Maximo en disco = MAX_BYTES * (BACKUP_COUNT + 1) por modelo.
_MAX_BYTES    = 512 * 1024   # 512 KB por archivo
_BACKUP_COUNT = 2            # + el archivo actual = 3 archivos → max ~1.5 MB por modelo

def setup_model_logger(model_name: str, log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{model_name}.log")

    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Handler para archivo con rotacion automatica
        fh = RotatingFileHandler(
            log_path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(fh)

        # Handler para consola
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(ch)

    return logger

#   ETAPAS A LOGUEAR:

# | Zona del pipeline        | Nivel     | Que loguear                                                     |
# | ------------------------ | --------- | --------------------------------------------------------------- |
# | `ModelLoader`            | `INFO`    | Modelo cargado con exito (nombre, formato, tipo)                |
# |                          | `ERROR`   | Fallo al cargar modelo (formato incompatible, ruta inexistente) |
# | `ConfigLoader` (JSON)    | `INFO`    | Config cargada correctamente                                    |
# |                          | `ERROR`   | Campo faltante o invalido (con Pydantic)                        |
# | `Preprocessor`           | `INFO`    | Transformacion aplicada, tamaño de imagen final                 |
# |                          | `WARNING` | Transformacion no compatible o parametro ignorado               |
# | `InferenceEngine`        | `INFO`    | Inicio/fin de inferencia, tiempo                                |
# |                          | `ERROR`   | Fallo en inferencia (shape incorrecto, dtype, etc)              |
# | `Unpacker / Adapter`     | `INFO`    | Shape esperado vs recibido                                      |
# |                          | `ERROR`   | No se puede desempaquetar la salida                             |
# | `Postprocessor`          | `INFO`    | Salida final transformada                                       |
# |                          | `ERROR`   | Fallo en NMS, umbral invalido, etc                              |


class PerfMeter:
    """
    Guarda los ultimos N tiempos y calcula fps promedio y p95.
    Pensado para benchmark en loop.
    Ventana chica (30) para que las metricas reflejen el estado actual rapidamente.
    """
    def __init__(self, window=30):
        self.window = window
        self.t_pre  = deque(maxlen=window)
        self.t_inf  = deque(maxlen=window)
        self.t_post = deque(maxlen=window)
        self.t_total= deque(maxlen=window)

    def reset(self) -> None:
        """Descarta todos los tiempos acumulados. Llamar al cambiar de modelo."""
        self.t_pre.clear()
        self.t_inf.clear()
        self.t_post.clear()
        self.t_total.clear()

    def push(self, pre_ms, inf_ms, post_ms, total_ms) -> None:
        self.t_pre.append(pre_ms)
        self.t_inf.append(inf_ms)
        self.t_post.append(post_ms)
        self.t_total.append(total_ms)

    def stats(self) -> Optional[Dict[str, Any]]:
        if not self.t_total:
            return None
        total = np.asarray(self.t_total, dtype=np.float32)

        avg_ms = float(total.mean())
        p95_ms = float(np.percentile(total, 95))
        fps_avg = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    
        pre_avg  = float(np.mean(np.asarray(self.t_pre,  dtype=np.float32))) if self.t_pre else 0.0
        inf_avg  = float(np.mean(np.asarray(self.t_inf,  dtype=np.float32))) if self.t_inf else 0.0
        post_avg = float(np.mean(np.asarray(self.t_post, dtype=np.float32))) if self.t_post else 0.0

        return {
            "avg_ms": avg_ms,
            "p95_ms": p95_ms,
            "fps_avg": fps_avg,
            "pre_avg_ms": pre_avg,
            "inf_avg_ms": inf_avg,
            "post_avg_ms": post_avg,
            "n": len(total),
        }


def run_warmup(predict_fn, dummy_input, runs: int, logger=None):
    for _ in range(runs):
        _ = predict_fn(dummy_input)
    if logger:
        logger.info(f"Warmup OK: {runs} runs")


def make_dummy_input(preprocess_fn, input_adapter, input_cfg):
    h0, w0 = input_cfg.height, input_cfg.width
    img = np.zeros((h0, w0, 3), dtype=np.uint8)

    img_prep = preprocess_fn(img)
    x = input_adapter(img_prep)
    return x