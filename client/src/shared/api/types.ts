// types.ts — tipos del frontend espejo del schema Pydantic ESTRICTO del backend
// (src/api/func/reader_pipeline/config_schema.py). Mantener sincronizado: si cambia
// el schema, cambia aca. El config es una union discriminada por model_type.

export type ModelType = 'detection' | 'classification' | 'segmentation';
export type Backend = 'onnxruntime' | 'tflite' | 'tensorflow' | 'pytorch';

// ── Input ────────────────────────────────────────────────────────────────────

export interface InputTensor {
  layout: 'HWC' | 'CHW' | 'NHWC' | 'NCHW';
  dtype: 'float32' | 'uint8' | 'int8';
  quantized: boolean;
}

export interface InputConfig {
  width: number;
  height: number;
  channels: number;
  normalize: boolean;
  mean: number[];
  std: number[];
  scale: boolean;
  letterbox: boolean;
  auto_pad_color: number[] | null;
  preserve_aspect_ratio: boolean | null;
  color_order: 'RGB' | 'BGR' | 'GRAY';
  input_str: InputTensor | null;
}

// ── Estructuras de tensor (una por tipo) ──────────────────────────────────────

export interface TensorDetection {
  box_format: 'xyxy' | 'cxcywh' | 'yxyx';
  coordinates: Record<string, number>;
  confidence_index: number;
  class_index: number;
  num_classes: number | null;
}

export interface TensorClassification {
  num_classes: number;
  output_format: 'logits' | 'probabilities';
  multi_label: boolean;
}

export interface TensorSegmentation {
  num_classes: number;
  output_format: 'argmax_map' | 'softmax_map';
  output_stride: number;
  resize_to_input: boolean;
  colormap: Record<number, number[]> | null; // {class_id: [R,G,B]}
}

// ── Output (uno por tipo) ─────────────────────────────────────────────────────

export interface AnchorConfig {
  min_level: number;
  max_level: number;
  num_scales: number;
  aspect_ratios: number[];
  anchor_scale: number;
  box_variance: number[];
  scores_activation: 'none' | 'sigmoid' | 'softmax';
}

export interface DetectionOutput {
  apply_conf_filter: boolean;
  confidence_threshold: number;
  apply_nms: boolean;
  top_k: number;
  nms_per_class: boolean;
  nms_threshold: number;
  tensor_structure: TensorDetection;
  pack_format: 'raw' | 'yolo_flat' | 'boxes_scores' | 'tflite_detpost' | 'anchor_deltas';
  anchor_config: AnchorConfig | null; // requerido si pack_format === 'anchor_deltas'
}

export interface ClassificationOutput {
  apply_softmax: boolean;
  apply_sigmoid: boolean;
  top_k: number;
  confidence_threshold: number;
  label_map: string[] | string | null;
  tensor_structure: TensorClassification;
  pack_format: 'softmax_out' | 'sigmoid_out' | 'logits_raw';
}

export interface SegmentationOutput {
  confidence_threshold: number;
  label_map: string[] | string | null;
  tensor_structure: TensorSegmentation;
  pack_format: 'argmax_map' | 'softmax_map' | 'binary_mask';
}

// ── Runtime ───────────────────────────────────────────────────────────────────

// Nota: anchors/box_variance del RuntimeShapes del backend tienen exclude=True
// (se generan al cargar, no van en el JSON), por eso NO aparecen aca.
export interface RuntimeShapes {
  input_width: number;
  input_height: number;
  channels: number;
  out_coords_space: 'normalized_0_1' | 'tensor_pixels';
}

export interface ThreadsConfig {
  intra_op: number | null;
  inter_op: number | null;
  num_threads: number | null;
}

export interface WarmupConfig {
  runs: number;
  enabled: boolean;
}

export interface OnnxRuntimeConfig {
  providers: string[];
  provider_options: Record<string, Record<string, unknown>>;
}

export interface TfliteRuntimeConfig {
  delegates: string[];
  delegate_options: Record<string, Record<string, unknown>>;
}

export interface RuntimeConfig {
  runtimeShapes: RuntimeShapes | null;
  backend: Backend;
  device: 'cpu' | 'gpu';
  threads: ThreadsConfig;
  onnx: OnnxRuntimeConfig | null;
  tflite: TfliteRuntimeConfig | null;
  warmup: WarmupConfig;
}

// ── Config raiz (union discriminada por model_type) ───────────────────────────

export interface DetectionModelConfig {
  model_type: 'detection';
  input: InputConfig;
  output: DetectionOutput;
  runtime: RuntimeConfig;
}

export interface ClassificationModelConfig {
  model_type: 'classification';
  input: InputConfig;
  output: ClassificationOutput;
  runtime: RuntimeConfig;
}

export interface SegmentationModelConfig {
  model_type: 'segmentation';
  input: InputConfig;
  output: SegmentationOutput;
  runtime: RuntimeConfig;
}

export type ModelConfig =
  | DetectionModelConfig
  | ClassificationModelConfig
  | SegmentationModelConfig;
