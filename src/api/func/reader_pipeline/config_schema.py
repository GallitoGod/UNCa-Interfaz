from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Union, Any

TYPE = Literal["detection", "classification", "segmentation"]
BACKEND = Literal["onnxruntime", "tflite", "tensorflow", "pytorch"]

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class InputTensor(BaseModel):
    layout: Literal["HWC", "CHW", "NHWC", "NCHW"] = "HWC"
    dtype: Literal["float32", "uint8", "int8"] = "float32"
    quantized: bool = False

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
    input_str: InputTensor = None

# ---------------------------------------------------------------------------
# Tensor structures (una por tipo de modelo)
# ---------------------------------------------------------------------------

class TensorDetection(BaseModel):
    box_format: Literal["xyxy", "cxcywh", "yxyx"] = "xyxy"
    coordinates: Dict[str, int] = {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
    confidence_index: int = 6
    class_index: int = 5
    num_classes: Optional[int] = None

class TensorClassification(BaseModel):
    num_classes: int
    output_format: Literal["logits", "probabilities"] = "logits"
    multi_label: bool = False  # True -> sigmoid por clase, False -> softmax (mutuamente exclusivo)

class TensorSegmentation(BaseModel):
    num_classes: int
    output_format: Literal["argmax_map", "softmax_map"] = "argmax_map"
    # argmax_map: el tensor ya es el indice de clase ganador por pixel (H x W)
    # softmax_map: el tensor tiene probabilidades por canal (C x H x W o H x W x C)
    output_stride: int = 1          # factor de reduccion respecto a la entrada (ej: 8, 16, 32)
    resize_to_input: bool = True    # hacer upsample de la mascara al tamano original
    colormap: Optional[Dict[int, List[int]]] = None  # {class_id: [R, G, B]}

# ---------------------------------------------------------------------------
# Output configs (una por tipo de modelo)
# ---------------------------------------------------------------------------

class DetectionOutput(BaseModel):
    apply_conf_filter: bool = True
    confidence_threshold: float = 0.5
    apply_nms: bool = False
    top_k: int = False
    nms_per_class: bool = False
    nms_threshold: float = 0.45
    tensor_structure: TensorDetection = Field(default_factory=TensorDetection)
    pack_format: Literal["raw", "yolo_flat", "boxes_scores", "tflite_detpost", "anchor_deltas"] = "raw"

class ClassificationOutput(BaseModel):
    apply_softmax: bool = True      # aplicar softmax sobre el vector de salida
    apply_sigmoid: bool = False     # usar en lugar de softmax cuando multi_label=True
    top_k: int = 1                  # cuantas predicciones devolver
    confidence_threshold: float = 0.5
    label_map: Optional[Union[List[str], str]] = None  # lista de nombres o path a .txt
    tensor_structure: TensorClassification
    pack_format: Literal["softmax_out", "sigmoid_out", "logits_raw"] = "softmax_out"

class SemanticSegmentationOutput(BaseModel):
    confidence_threshold: float = 0.5  # umbral minimo para reportar una clase (en softmax_map)
    label_map: Optional[Union[List[str], str]] = None
    tensor_structure: TensorSegmentation
    pack_format: Literal["argmax_map", "softmax_map", "binary_mask"] = "argmax_map"

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class RuntimeShapes(BaseModel):
    input_width: int = 0
    input_height: int = 0
    orig_width: int = 0
    orig_height: int = 0
    metadata_letter: Dict[str, Union[float, bool]] = Field(default_factory=lambda: {
        "scale": 1.0,
        "pad_left": 0.0,
        "pad_top": 0.0,
        "letterbox_used": False
    })
    channels: int = 3
    out_coords_space: Literal["normalized_0_1", "tensor_pixels"] = "normalized_0_1"

class ThreadsConfig(BaseModel):
    intra_op: Optional[int] = None   # ONNXRuntime
    inter_op: Optional[int] = None   # ONNXRuntime
    num_threads: Optional[int] = None  # TFLite / general

class WarmupConfig(BaseModel):
    runs: int = 0
    enabled: bool = True

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

# Alias para type hints en pipelines que aceptan cualquier tipo de output
AnyOutputConfig = Union[DetectionOutput, ClassificationOutput, SemanticSegmentationOutput]

# ---------------------------------------------------------------------------
# Config raiz
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    model_type: TYPE
    input: InputConfig
    output: Union[DetectionOutput, ClassificationOutput, SemanticSegmentationOutput]
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

'''
    Hay muchos cambios en nombres de configuraciones, se van a dar varios fallos en todo el sistema
    cuando se empiece a testear. Simplemente hay que poner los nombres actuales.

    NOTA: InputDetection fue renombrado a InputTensor (el concepto es generico, no es solo deteccion).
    El campo input_str de InputConfig mantiene el mismo nombre por compatibilidad.

    Pydantic resuelve el Union de output intentando cada tipo en orden: Detection -> Classification -> Segmentation.
    Si hay ambiguedad en un JSON, mover el tipo mas restrictivo primero en el Union.
'''
