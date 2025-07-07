from typing import Callable
import numpy as np

def generate_box_converter(fmt: str, coords: dict) -> Callable[[list], list]:
    if fmt == "xyxy":
        return lambda row: [
            row[coords["x1"]],
            row[coords["y1"]],
            row[coords["x2"]],
            row[coords["y2"]],
        ]
    elif fmt == "cxcywh":
        return lambda row: [
            row[coords["cx"]] - row[coords["w"]] / 2,
            row[coords["cy"]] - row[coords["h"]] / 2,
            row[coords["cx"]] + row[coords["w"]] / 2,
            row[coords["cy"]] + row[coords["h"]] / 2,
        ]
    elif fmt == "yxyx":
        return lambda row: [
            row[coords["x1"]],
            row[coords["y1"]],
            row[coords["x2"]],
            row[coords["y2"]],
        ]
    else:
        raise ValueError(f"Formato desconocido: {fmt}")


def generate_layout_converter(layout: str):
    """
    layout: "HWC", "CHW", "NHWC", "NCHW"
    Devuelve una funcion que adapta la entrada a ese formato
    """
    def to_layout(img: np.ndarray) -> np.ndarray:
        if layout == "CHW":
            return np.transpose(img, (2, 0, 1))
        elif layout == "NHWC":
            return img[np.newaxis, ...]
        elif layout == "NCHW":
            img = np.transpose(img, (2, 0, 1))
            return img[np.newaxis, ...]
        elif layout == "HWC":
            return img
        else:
            raise ValueError(f"Formato de layout desconocido: {layout}")
    return to_layout
