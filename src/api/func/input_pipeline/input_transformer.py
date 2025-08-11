from api.func.reader_pipeline.config_schema import InputConfig, RuntimeSession
from typing import Callable
import numpy as np
import cv2


def build_letterbox(input_width, input_height, pad_color):
    # Imagen base: W=1920, H=1080
    h, w = 1080, 1920
    scale = min(input_width / w, input_height / h)
    nw, nh = int(w * scale), int(h * scale)
    pad_w = input_width - nw
    pad_h = input_height - nh
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    def letterbox(img):
        resized = cv2.resize(img, (nw, nh))
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color
        )
        return padded
    transform_info = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "letterbox_used": True
    }

    return letterbox, transform_info

def build_preprocessor(config: InputConfig, runtime: RuntimeSession) -> Callable[[np.ndarray], np.ndarray]:
    try:
        steps = []
        if config.letterbox and config.preserve_aspect_ratio:
            letterbox, transform_info = build_letterbox(config.width, config.height, config.auto_pad_color)
            steps.append(letterbox)
            runtime.metadata_letter = transform_info
        else:
            steps.append(lambda img: cv2.resize(img, (config.width, config.height)))
            runtime.metadata_letter = {
                "scale": 1.0,
                "pad_left": 0.0,
                "pad_top": 0.0,
                "letterbox_used": False
            }

        if config.scale:
            steps.append(lambda img: img.astype(np.float32) / 255.0)

        if config.normalize:
            mean = np.array(config.mean).reshape(1, 1, -1)
            std = np.array(config.std).reshape(1, 1, -1)
            if np.any(std == 0):
                raise ValueError("std no puede contener ceros para la normalizacion")
            steps.append(lambda img: (img - mean) / std)

        def preprocess(img):
            for step in steps:
                img = step(img)
            return img
        
        return preprocess
    except Exception as e:
        raise ValueError(e) from e