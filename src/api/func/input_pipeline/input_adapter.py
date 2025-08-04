from api.func.general.config_schema import InputConfig
import numpy as np

def generate_input_adapter(input_config: InputConfig):
    tensor_cfg = input_config.input_tensor or None
    color_order = input_config.color_order or "RGB"
    layout_converter = generate_layout_converter(tensor_cfg.layout) if tensor_cfg else lambda x: x
    dtype = tensor_cfg.dtype if tensor_cfg else "float32"

    if color_order == "BGR":
        def adapter_fn_in(img: np.ndarray) -> np.ndarray:
            img = img[..., ::-1]           # invierte canales
            img = layout_converter(img)    # reordenar layout
            return img.astype(dtype)       # tipo final
    else:
        def adapter_fn_in(img: np.ndarray) -> np.ndarray:
            img = layout_converter(img)
            return img.astype(dtype)

    return adapter_fn_in

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