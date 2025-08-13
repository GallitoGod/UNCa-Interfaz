from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Union

class InputTensorConfig(BaseModel):
    layout: Literal["HWC", "CHW", "NHWC", "NCHW"] = "HWC"
    dtype: Literal["float32", "uint8", "int8"] = "float32"
    quantized: bool = False

class OutputTensorConfig(BaseModel):
    #layout: Literal["HWC", "CHW", "NHWC", "NCHW"] = "HWC"  No es necesario 
    #dtype: Literal["float32", "int8"] = "float32"          No es necesario 
    output_format: Literal["raw", "multihead", "softmax"] = "raw"

class TensorStructure(BaseModel):
    box_format: Literal["xyxy", "cxcywh", "yxyx"]
    coordinates: Dict[str, int]
    confidence_index: int
    class_index: int
    num_classes: Optional[int]

class OutputConfig(BaseModel):
    confidence_threshold: float = 0.5
    nms_threshold: Optional[float] = 0.45
    apply_nms: bool = Field(default=False)
    tensor_structure: Optional[TensorStructure]
    output_tensor: Optional[OutputTensorConfig] = None

class InputConfig(BaseModel):
    width: int
    height: int
    channels: int
    normalize: bool = True
    mean: List[float] = [0.0, 0.0, 0.0]
    std: List[float] = [1.0, 1.0, 1.0]
    scale: bool = True
    letterbox: bool = Field(default=False)
    auto_pad_color: Optional[List[int]] = [114, 114, 114]
    preserve_aspect_ratio: Optional[bool] = True
    color_order: Optional[str] = "RGB"
    input_tensor: Optional[InputTensorConfig] = None 

class RuntimeSession(BaseModel):
    metadata_letter: Optional[Dict[str, Union[float, bool]]] = {
    "scale": 1.0,
    "pad_left": 0.0,
    "pad_top": 0.0,
    "letterbox_used": False
    }
    channels: int = 3
    #DEVICE: Optional[Literal["CPU", "GPU", "TPU", "NPU"]] = "CPU" <--- TENER EN CUENTA EN EL FUTURO
    # Mas adelante ire agregando datos mutables que sean necesarios

class ModelConfig(BaseModel):
    model_type: Literal["detection", "classification", "segmentation"]
    input: InputConfig
    output: OutputConfig
    runtime: RuntimeSession = Field(default_factory=RuntimeSession)
