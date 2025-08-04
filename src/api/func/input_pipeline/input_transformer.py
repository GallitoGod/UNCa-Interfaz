from api.func.general.config_schema import InputConfig
from typing import Callable
import numpy as np
import cv2


def build_letterbox(input_width, input_height, pad_color):
    def letterbox(img):
        h, w = img.shape[:2]
        scale = min(input_width / w, input_height / h)
        nw, nh = int(w * scale), int(h * scale)
        pad_w = input_width - nw
        pad_h = input_height - nh
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        resized = cv2.resize(img, (nw, nh))
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color
        )
        return padded, scale, pad_left, pad_top

    return letterbox

def buildPreprocessor(config: InputConfig) -> Callable[[np.ndarray], np.ndarray]:
    try:
        steps = []
        transform_info = {}
        if config.letterbox:
            letterbox = build_letterbox(config.width, config.height, config.auto_pad_color)

            def letterbox_wrapper(img):
                padded, scale, pad_left, pad_top = letterbox(img)
                transform_info['scale'] = scale
                transform_info['pad_left'] = pad_left
                transform_info['pad_top'] = pad_top
                transform_info['used_letterbox'] = True
                return padded

            steps.append(letterbox_wrapper)
        else:
            steps.append(lambda img: cv2.resize(img, (config.width, config.height)))

        if config.scale:
            steps.append(lambda img: img.astype(np.float32) / 255.0)

        if config.normalize:
            mean = np.array(config.mean).reshape(1, 1, -1)
            std = np.array(config.std).reshape(1, 1, -1)
            steps.append(lambda img: (img - mean) / std)

        def preprocess(img):
            for step in steps:
                img = step(img)
            return img

        return preprocess, transform_info
    except Exception as e:
        raise ValueError(f"Error: {e}") from e