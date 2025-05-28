from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union

# --- NUEVO ---
class TensorStructure(BaseModel):
    box_format: Literal["xyxy", "cxcywh"]
    box_indices: List[int] = Field(..., min_items=4, max_items=4)
    confidence_index: int
    class_index: int
    num_classes: Optional[int]

class OutputConfig(BaseModel):
    confidence_threshold: float = 0.5
    nms_threshold: Optional[float] = 0.45
    apply_nms: bool = Field(default=False)

    # --- NUEVO ---
    tensor_structure: Optional[TensorStructure] = None
    output_dtype: Optional[str] = "float32"
    output_format: Optional[Literal["NXC", "NCX"]] = "NXC"

class InputConfig(BaseModel):
    width: int
    height: int
    channels: int
    normalize: bool = True
    mean: List[float] = [0.0, 0.0, 0.0]
    std: List[float] = [1.0, 1.0, 1.0]
    letterbox: bool = Field(default=False)
    auto_pad_color: Optional[List[int]] = [114, 114, 114]
    preserve_aspect_ratio: Optional[bool] = True

    # --- NUEVO ---
    data_format: Optional[Literal["NHWC", "NCHW"]] = "NHWC"
    dtype: Optional[str] = "float32"
    scale: Optional[float] = 1.0

class ModelConfig(BaseModel):
    model_type: Literal["detection", "classification", "segmentation"]
    input: InputConfig
    output: OutputConfig
