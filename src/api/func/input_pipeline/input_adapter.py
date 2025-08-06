import numpy as np
from ..reader_pipeline import InputConfig, RuntimeSession

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

def generate_input_adapter(input_config: InputConfig, runtime: RuntimeSession):
    tensor_cfg = input_config.input_tensor or None
    color_order = input_config.color_order or "RGB"
    layout_converter = generate_layout_converter(tensor_cfg.layout) if tensor_cfg else lambda x: x
    dtype = tensor_cfg.dtype if tensor_cfg else "float32"
    channels = input_config.channels or 3
    
    if channels == 1:
        color_order = "GRAY"
        runtime.channels = 1
    if channels == 3:
        pass
    else:
        raise ValueError(f"Canal invalido: {channels}. Solo 1 (GRAY) o 3 (RGB/BGR) son soportados.")

    if color_order == "BGR":
        def adapter_fn_in(img: np.ndarray) -> np.ndarray:
            img = layout_converter(img)    # reordenar layout
            img = img[..., ::-1]           # invertir canales
            return img.astype(dtype)       # tipo final
    if color_order == "GRAY":
        def adapter_fn_in(img: np.ndarray) -> np.ndarray:
            img = layout_converter(img)    
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            return img.astype(dtype)       
    if color_order == "RGB":
        def adapter_fn_in(img: np.ndarray) -> np.ndarray:
            img = layout_converter(img)
            return img.astype(dtype)
    else:
        raise ValueError(f"Color_order invalido: {color_order}. Solo RGB, BGR o GRAY son soportados.")
    
    return adapter_fn_in
