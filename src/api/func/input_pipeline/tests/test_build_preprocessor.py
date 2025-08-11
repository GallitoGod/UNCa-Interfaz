import numpy as np
import pytest
from api.func.input_pipeline.input_transformer import build_preprocessor
from api.func.reader_pipeline.config_schema import InputConfig


class DummyRuntime:
    def __init__(self):
        self.metadata_letter = {
            "scale": 1.0,
            "pad_left": 0.0,
            "pad_top": 0.0,
            "letterbox_used": False
        }


def test_preprocessor_with_letterbox_and_scaling_and_normalization():
    cfg = InputConfig(
        width=640,                              #width: int
        height=360,                             #height: int
        channels= 3,                            #channels: int
        normalize=True,                         #normalize: bool = True                  
        mean=[0.5, 0.5, 0.5],                   #mean: List[float] = [0.0, 0.0, 0.0]
        std=[0.5, 0.5, 0.5],                    #std: List[float] = [1.0, 1.0, 1.0]
        scale=True,                             #scale: bool = True
        letterbox=True,                         #letterbox: bool = Field(default=False)                   
        auto_pad_color=(0, 0, 0),               #auto_pad_color: Optional[List[int]] = [114, 114, 114]
        preserve_aspect_ratio=True,             #preserve_aspect_ratio: Optional[bool] = True               
        color_order="RGB",                      #color_order: Optional[Literal["RGB", "BGR", "GRAY"]] = "RGB"
        input_tensor= None                      #input_tensor: Optional[InputTensorConfig] = None                       
    )                                           
    runtime = DummyRuntime()
    preprocess_fn = build_preprocessor(cfg, runtime)

    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    out = preprocess_fn(img)

    # Tamaño final debe coincidir con config
    assert out.shape == (360, 640, 3)

    # Debe estar normalizado: valor medio ~ 1.0 para una imagen blanca
    assert np.allclose(out, (1.0 - 0.5) / 0.5, atol=1e-6)

    # metadata_letter debe haberse seteado
    assert runtime.metadata_letter["letterbox_used"] is True


def test_preprocessor_without_letterbox():
    cfg = InputConfig(
        width=320,
        height=240,
        channels=3,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        scale=True,
        letterbox=False,
        auto_pad_color=(0, 0, 0),
        preserve_aspect_ratio=False,
        color_order="RGB",
        input_tensor=None
    )

    runtime = DummyRuntime()
    preprocess_fn = build_preprocessor(cfg, runtime)

    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    out = preprocess_fn(img)

    assert out.shape == (240, 320, 3)
    assert runtime.metadata_letter["letterbox_used"] is False


def test_preprocessor_invalid_config():
    # Provocar un error en normalizacion (std=0 genera division por cero)
    cfg = InputConfig(
        width=320,
        height=240,
        channels= 3,                           
        normalize=True,                                      
        mean=[0, 0, 0],                
        std=[0, 0, 0],                  
        scale=False,
        letterbox=False,
        auto_pad_color=(0, 0, 0),
        preserve_aspect_ratio=False,
        color_order="RGB",                      
        input_tensor= None 
    )
    runtime = DummyRuntime()

    with pytest.raises(ValueError):
        build_preprocessor(cfg, runtime)