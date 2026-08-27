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
    Guarda los ultimos `window` tiempos y calcula fps promedio y p95.
    Pensado para benchmark en loop. El controller usa window=300 (~10s a 30fps):
    suaviza el HUD a costa de inercia; bajar la ventana si se quiere reaccion rapida.
    """
    def __init__(self, window=30):
        self.window = window
        self.t_pre  = deque(maxlen=window)
        self.t_inf  = deque(maxlen=window)
        self.t_post = deque(maxlen=window)
        # Anotado + re-encode JPEG del frame compuesto (paso 3, 2026-08-26). Va en su
        # PROPIO bucket a proposito: si cayera dentro de post_ms quedaria invisible, y
        # es justo el numero que hace falta para defender la decision de mudar el
        # render al backend. No entra en t_total: total mide el pipeline de inferencia.
        self.t_draw = deque(maxlen=window)
        # Tracking + suavizado (Tier B). Bucket propio por el mismo motivo que
        # t_draw: es un costo OPCIONAL que el usuario prende, y hay que poder
        # mirarlo aislado para decidir si vale lo que cuesta. Tampoco entra en
        # t_total, que mide el pipeline de inferencia.
        self.t_track = deque(maxlen=window)
        self.t_total= deque(maxlen=window)

    def reset(self) -> None:
        """Descarta todos los tiempos acumulados. Llamar al cambiar de modelo."""
        self.t_pre.clear()
        self.t_inf.clear()
        self.t_post.clear()
        self.t_draw.clear()
        self.t_track.clear()
        self.t_total.clear()

    def push_draw(self, draw_ms) -> None:
        """Tiempo de composicion+encode de UN frame. Lo llama el controller al renderizar."""
        self.t_draw.append(draw_ms)

    def push_track(self, track_ms) -> None:
        """Tiempo de tracking+suavizado de UN frame. Lo empuja el handler del WS,
        que es donde vive la memoria de sesion (ver render/session.py)."""
        self.t_track.append(track_ms)

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
        draw_avg = float(np.mean(np.asarray(self.t_draw, dtype=np.float32))) if self.t_draw else 0.0
        track_avg = float(np.mean(np.asarray(self.t_track, dtype=np.float32))) if self.t_track else 0.0

        return {
            "avg_ms": avg_ms,            # pipeline de inferencia (pre+inf+post), SIN el dibujo
            "p95_ms": p95_ms,
            "fps_avg": fps_avg,
            "pre_avg_ms": pre_avg,
            "inf_avg_ms": inf_avg,
            "post_avg_ms": post_avg,
            "draw_avg_ms": draw_avg,     # anotado + re-encode (0 si la tarea no dibuja)
            "track_avg_ms": track_avg,   # tracking + suavizado (0 si estan apagados)
            # Lo que realmente cuesta un frame de punta a punta en el backend, y por
            # eso SUMA todos los costos opcionales: el cliente lo muestra como "Total
            # por frame". Si el tracking quedara afuera, prenderlo haria que el total
            # mostrado dejara de coincidir con la realidad. El nombre quedo del paso 3
            # (cuando el dibujo era el unico extra); se conserva porque es contrato
            # con el cliente.
            "avg_with_draw_ms": avg_ms + draw_avg + track_avg,
            "n": len(total),
        }


def run_warmup(predict_fn, dummy_input, runs: int, logger=None):
    for _ in range(runs):
        _ = predict_fn(dummy_input)
    if logger:
        logger.info(f"Warmup OK: {runs} runs")


def make_dummy_input(preprocess_fn, input_adapter, input_cfg):
    """input_cfg debe ser un InputConfig (config.input), NO el ModelConfig raiz."""
    h0, w0 = input_cfg.height, input_cfg.width
    img = np.zeros((h0, w0, 3), dtype=np.uint8)

    # preprocess_fn devuelve (tensor, meta); para el warmup solo importa el tensor
    img_prep, _meta = preprocess_fn(img)
    x = input_adapter(img_prep)
    return x