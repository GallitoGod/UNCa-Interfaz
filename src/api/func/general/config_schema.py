from pydantic import BaseModel, Field
from typing import List, Optional

class NormalizeConfig(BaseModel):
    enabled: bool
    mean: List[float] = Field(default=[0.0, 0.0, 0.0])
    std: List[float] = Field(default=[1.0, 1.0, 1.0])

class PostprocessingConfig(BaseModel):
    nms: Optional[bool] = False
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: Optional[float] = Field(default=0.45, ge=0, le=1)

class ModelConfig(BaseModel):
    input_shape: List[int]  
    keep_aspect_ratio: Optional[bool] = True
    normalize: NormalizeConfig
    color_format: Optional[str] = "RGB"
    model_type: str  
    postprocessing: Optional[PostprocessingConfig]


