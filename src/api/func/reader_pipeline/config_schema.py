from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Union, Any

TYPE = Literal["detection", "classification", "segmentation"]
BACKEND = Literal["onnxruntime", "tflite", "tensorflow"]

class InputDetection(BaseModel):
    layout: Literal["HWC", "CHW", "NHWC", "NCHW"] = "HWC"
    dtype: Literal["float32", "uint8", "int8"] = "float32"
    quantized: bool = False

class TensorDetection(BaseModel):
    '''
        Tengo que darle valores por defecto para que en OutputConfig pueda usar 
    Filed(default=TensorDetection).
    '''
    box_format: Literal["xyxy", "cxcywh", "yxyx"]
    coordinates: Dict[str, int]
    confidence_index: int
    class_index: int
    num_classes: Optional[int]

class OutputConfig(BaseModel):
    apply_conf_filter: bool = True
    confidence_threshold: float = 0.5
    apply_nms: bool = False
    top_k: int = False
    nms_per_class: bool = False
    nms_threshold: float = 0.45
    tensor_structure: Optional[TensorDetection]
    pack_format: Literal["raw", "yolo_flat", "boxes_scores", "tflite_detpost", "anchor_deltas"] = "raw"

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
    color_order: Literal["RGB", "BGR", "GRAY"] = "RGB"
    input_config: InputDetection = None # <---  Aca era 'input_tensor', lo cual no es correcto.

class RuntimeShapes(BaseModel):
    input_width: int = 0
    input_height: int = 0
    orig_width: int = 0
    orig_height: int = 0
    metadata_letter: Optional[Dict[str, Union[float, bool]]] = Field(default_factory= lambda: {
        "scale": 1.0,
        "pad_left": 0.0,
        "pad_top": 0.0,
        "letterbox_used": False
    })
    channels: int = 3
    out_coords_space: Literal["normalized_0_1", "tensor_pixels"] = "normalized_0_1"

class ThreadsConfig(BaseModel):
    # ONNXRuntime
    intra_op: Optional[int] = None
    inter_op: Optional[int] = None
    # TFLite / general
    num_threads: Optional[int] = None

class WarmupConfig(BaseModel):
    runs: int = 0  # 0 = off
    enabled: bool = True  # opcional, por si se quiere apagar aunque runs>0

class OnnxRuntimeConfig(BaseModel):
    providers: List[str] = Field(default_factory=list)
    provider_options: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

class TfliteRuntimeConfig(BaseModel):
    delegates: List[str] = Field(default_factory=list)
    delegate_options: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

class RuntimeConfig(BaseModel):
    runtimeShapes: Optional[RuntimeShapes] = Field(default_factory=RuntimeShapes)
    backend: BACKEND = "onnxruntime"
    device: Literal["cpu", "gpu"] = "cpu"
    threads: ThreadsConfig = Field(default_factory=ThreadsConfig)

    onnx: Optional[OnnxRuntimeConfig] = None
    tflite: Optional[TfliteRuntimeConfig] = None

    warmup: WarmupConfig = Field(default_factory=WarmupConfig)

class ModelConfig(BaseModel):
    model_type: TYPE
    input: InputConfig
    output: OutputConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

'''
    Hay muchos cambios en nombres de configuraciones, se van a dar varios fallos en todo el sistema 
cuando se empiece a testear. Simplemente hay que poner los nombres actuales.
'''