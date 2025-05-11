from pydantic import BaseModel, Field
from typing import List, Optional

class OutputConfig(BaseModel):
    confidence_threshold: float = 0.5
    nms_threshold: Optional[float] = 0.45
    apply_nms: bool = Field(default=False)

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

class ModelConfig(BaseModel):
    model_type: str  
    input: InputConfig
    output: OutputConfig


