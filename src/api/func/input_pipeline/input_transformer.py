from api.func.reader_pipeline.config_schema import InputConfig, RuntimeConfig
from typing import Callable
import numpy as np
import cv2

def build_letterbox(pad_color, runtime):
    """
    Letterbox preprocessor builder.
    - Lee input_width/input_height desde runtime.runtimeShapes (source of truth).
    - Calcula scale/pads cuando cambia el tamaño del frame (cache por shape).
    - Aplica resize + padding y actualiza runtime.runtimeShapes.metadata_letter.
    """

    shapes = runtime.runtimeShapes
    if shapes is None:
        raise ValueError("RuntimeConfig.runtimeShapes no puede ser None para letterbox")

    iw = int(shapes.input_width)
    ih = int(shapes.input_height)

    last_wh = None
    cached = None  # (scale, nw, nh, pad_left, pad_right, pad_top, pad_bottom)

    def letterbox(img):
        nonlocal last_wh, cached

        h, w = img.shape[:2]

        shapes.orig_width = int(w)
        shapes.orig_height = int(h)

        if last_wh != (w, h) or cached is None:
            scale = min(iw / w, ih / h)
            nw, nh = int(w * scale), int(h * scale)

            pad_w = iw - nw
            pad_h = ih - nh

            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            cached = (scale, nw, nh, pad_left, pad_right, pad_top, pad_bottom)
            last_wh = (w, h)

            md = shapes.metadata_letter
            md["scale"] = float(scale)
            md["pad_left"] = float(pad_left)
            md["pad_top"] = float(pad_top)
            md["letterbox_used"] = True

        scale, nw, nh, pad_left, pad_right, pad_top, pad_bottom = cached

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

        resized = cv2.resize(img, (nw, nh), interpolation=interp)

        out = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color
        )

        return out

    return letterbox

def build_preprocessor(config: InputConfig, runtime: RuntimeConfig) -> Callable[[np.ndarray], np.ndarray]:
    try:
        runtime.runtimeShapes.input_width  = int(config.width)
        runtime.runtimeShapes.input_height = int(config.height)

        steps = []
        if config.letterbox and config.preserve_aspect_ratio:
            letterbox = build_letterbox(config.auto_pad_color, runtime)
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
            runtime.runtimeShapes.orig_width  = int(w)
            runtime.runtimeShapes.orig_height = int(h)
            if not (config.letterbox and config.preserve_aspect_ratio):
                runtime.runtimeShapes.metadata_letter = {
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
