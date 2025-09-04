from api.func.reader_pipeline.config_schema import InputConfig, RuntimeSession
from typing import Callable
import numpy as np
import cv2

def build_letterbox(input_width, input_height, pad_color, runtime: RuntimeSession):
    # ¡OJO! Calculamos todo por frame, usando img.shape, no runtime.width/height.
    def letterbox(img):
        h, w = img.shape[:2]
        scale = min(input_width / w, input_height / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh))

        pad_w = input_width - nw
        pad_h = input_height - nh
        pad_left  = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top   = pad_h // 2
        pad_bottom= pad_h - pad_top

        out = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=pad_color
        )

        # Actualizamos metadata de letterbox y tamaño original del frame
        runtime.metadata_letter = {
            "scale": float(scale),
            "pad_left": float(pad_left),
            "pad_top":  float(pad_top),
            "letterbox_used": True
        }
        runtime.orig_width  = int(w)
        runtime.orig_height = int(h)
        return out

    return letterbox

def build_preprocessor(config: InputConfig, runtime: RuntimeSession) -> Callable[[np.ndarray], np.ndarray]:
    try:
        runtime.input_width  = int(config.width)
        runtime.input_height = int(config.height)

        steps = []
        if config.letterbox and config.preserve_aspect_ratio:
            letterbox = build_letterbox(config.width, config.height, config.auto_pad_color, runtime)
            steps.append(letterbox)
        else:
            steps.append(lambda img: cv2.resize(img, (config.width, config.height)))

        if config.scale:
            steps.append(lambda img: img.astype(np.float32) / 255.0)

        if config.normalize:
            mean = np.array(config.mean).reshape(1, 1, -1)
            std  = np.array(config.std).reshape(1, 1, -1)
            if np.any(std == 0):
                raise ValueError("std no puede contener ceros para la normalizacion")
            steps.append(lambda img: (img - mean) / std)

        def preprocess(img):
            h, w = img.shape[:2]
            runtime.orig_width  = int(w)
            runtime.orig_height = int(h)
            if not (config.letterbox and config.preserve_aspect_ratio):
                runtime.metadata_letter = {
                    "scale": 1.0,
                    "pad_left": 0.0,
                    "pad_top":  0.0,
                    "letterbox_used": False
                }
            for step in steps:
                img = step(img)
            return img

        return preprocess
    except Exception as e:
        raise ValueError(e) from e
