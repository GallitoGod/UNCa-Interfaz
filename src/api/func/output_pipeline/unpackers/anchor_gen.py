# api/func/output_pipeline/unpackers/anchor_gen.py
from __future__ import annotations
import numpy as np


def generate_efficientdet_anchors(image_h: int, image_w: int, cfg) -> np.ndarray:
    """
    Genera la tabla de anchors estilo EfficientDet/automl (tambien valida para
    SSD multiscale de TF OD API con otros parametros).

    Orden de aplanado (debe coincidir con el head del modelo):
      niveles (min->max) > posiciones (fila-major) > configs (octava externa, aspect interno)

    Devuelve (N, 4) float32 [ay, ax, ah, aw] (centro y tamano) NORMALIZADOS [0..1].
    Para input 320x320 con defaults: N = (40^2+20^2+10^2+5^2+3^2)*9 = 19206.
    """
    levels = []
    for level in range(cfg.min_level, cfg.max_level + 1):
        stride = 2 ** level
        # SAME padding: las feature maps se reducen con ceil
        feat_h = int(np.ceil(image_h / stride))
        feat_w = int(np.ceil(image_w / stride))

        cy = (np.arange(feat_h, dtype=np.float32) + 0.5) * stride / image_h
        cx = (np.arange(feat_w, dtype=np.float32) + 0.5) * stride / image_w
        cyg, cxg = np.meshgrid(cy, cx, indexing="ij")          # (fh, fw)

        # (num_scales * num_aspects, 2) -> [ah, aw] por config
        per_config = []
        for octave in range(cfg.num_scales):
            scale = 2.0 ** (octave / cfg.num_scales)
            for aspect in cfg.aspect_ratios:
                sq = float(np.sqrt(aspect))
                aw = cfg.anchor_scale * stride * scale * sq / image_w
                ah = cfg.anchor_scale * stride * scale / sq / image_h
                per_config.append((ah, aw))
        per_config = np.asarray(per_config, dtype=np.float32)  # (A, 2)
        A = per_config.shape[0]

        n_loc = feat_h * feat_w
        ay = np.repeat(cyg.reshape(-1, 1), A, axis=1)          # (n_loc, A)
        ax = np.repeat(cxg.reshape(-1, 1), A, axis=1)
        ah = np.broadcast_to(per_config[:, 0], (n_loc, A))
        aw = np.broadcast_to(per_config[:, 1], (n_loc, A))

        levels.append(np.stack([ay, ax, ah, aw], axis=-1).reshape(-1, 4))

    return np.concatenate(levels, axis=0).astype(np.float32, copy=False)
