from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Union

class InputTensorConfig(BaseModel):
    layout: Literal["HWC", "CHW", "NHWC", "NCHW"] = "HWC"
    dtype: Literal["float32", "uint8", "int8"] = "float32"
    quantized: bool = False

class TensorStructure(BaseModel):
    box_format: Literal["xyxy", "cxcywh", "yxyx"]
    coordinates: Dict[str, int]
    confidence_index: int
    class_index: int
    num_classes: Optional[int]

class OutputConfig(BaseModel):
    apply_conf_filter: bool = True
    confidence_threshold: float = 0.5
    apply_nms: bool = False
    top_k: bool = False
    nms_per_class: bool = False
    nms_threshold: float = 0.45
    tensor_structure: Optional[TensorStructure]
    pack_format: Literal["raw", "yolo_flat", "boxes_scores", "tflite_detpost"] = "raw"

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
    input_tensor: InputTensorConfig = None 

class RuntimeSession(BaseModel):
    input_width: 0
    input_height: 0
    orig_width: int = 0
    orig_height: int = 0
    metadata_letter: Optional[Dict[str, Union[float, bool]]] = {
    "scale": 1.0,
    "pad_left": 0.0,
    "pad_top": 0.0,
    "letterbox_used": False
    }
    channels: int = 3
    out_coords_space: Literal["normalized_0_1", "tensor_pixels"] = "normalized_0_1"
    #DEVICE: Optional[Literal["CPU", "GPU", "TPU", "NPU"]] = "CPU" <--- TENER EN CUENTA EN EL FUTURO

class ModelConfig(BaseModel):
    model_type: Literal["detection", "classification", "segmentation"]
    input: InputConfig
    output: OutputConfig
    runtime: RuntimeSession = Field(default_factory=RuntimeSession)
