# logger.py
import logging
from collections import deque
import numpy as np
import os

def setup_model_logger(model_name: str, log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{model_name}.log")

    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)  # Captura todo, desde DEBUG

    if not logger.handlers:
        # Handler para archivo
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

        # Handler para consola
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(
            "%(levelname)s - %(message)s"
        )
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)

    return logger

#   ETAPAS A LOGUEAR:

# | Zona del pipeline        | Nivel     | Que loguear                                                     |
# | ------------------------ | --------- | --------------------------------------------------------------- |
# | `ModelLoader`            | `INFO`    | Modelo cargado con éxito (nombre, formato, tipo)                |
# |                          | `ERROR`   | Fallo al cargar modelo (formato incompatible, ruta inexistente) |
# | `ConfigLoader` (JSON)    | `INFO`    | Config cargada correctamente                                    |
# |                          | `ERROR`   | Campo faltante o inválido (con Pydantic)                        |
# | `Preprocessor`           | `INFO`    | Transformación aplicada, tamaño de imagen final                 |
# |                          | `WARNING` | Transformación no compatible o parámetro ignorado               |
# | `InferenceEngine`        | `INFO`    | Inicio/fin de inferencia, tiempo                                |
# |                          | `ERROR`   | Fallo en inferencia (shape incorrecto, dtype, etc)              |
# | `Unpacker / Adapter`     | `INFO`    | Shape esperado vs recibido                                      |
# |                          | `ERROR`   | No se puede desempaquetar la salida                             |
# | `Postprocessor`          | `INFO`    | Salida final transformada                                       |
# |                          | `ERROR`   | Fallo en NMS, umbral inválido, etc                              |


class PerfMeter:
    """
    Guarda los ultimos N tiempos y calcula fps promedio y p95.
    Pensado para benchmark en loop.
    """
    def __init__(self, window=300):
        self.window = window
        self.t_pre  = deque(maxlen=window)
        self.t_inf  = deque(maxlen=window)
        self.t_post = deque(maxlen=window)
        self.t_total= deque(maxlen=window)

    def push(self, pre_ms, inf_ms, post_ms, total_ms) -> None:
        self.t_pre.append(pre_ms)
        self.t_inf.append(inf_ms)
        self.t_post.append(post_ms)
        self.t_total.append(total_ms)

    def stats(self) -> dict | None:
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