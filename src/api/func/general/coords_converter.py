from typing import Callable

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
