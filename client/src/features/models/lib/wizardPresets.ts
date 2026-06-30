// wizardPresets.ts — helpers puros del wizard: presets y derivaciones.
// Sin React ni store: solo entrada -> salida, faciles de razonar y testear.
//
// Tres familias:
//   1. Normalizacion: preset <-> (scale, normalize, mean, std).
//   2. pack_format: formato -> (tensor_structure, out_coords_space). Constantes fijadas
//      contra el orden de columnas de cada unpacker (output_pipeline/unpackers/*.py) y
//      contra los configs reales (efficientdet-lite0/2 para anchor_deltas).
//   3. device: dispositivo -> providers (ONNX) / delegates (TFLite).

import type {
  Backend,
  DetectionOutput,
  InputConfig,
  TensorDetection,
} from '@/shared/api/types';

// ── 1. Normalizacion ───────────────────────────────────────────────────────────

export type NormPreset = 'none' | 'scale01' | 'imagenet' | 'custom';

export const NORM_PRESET_LABELS: Record<NormPreset, string> = {
  none: 'Ninguna',
  scale01: 'Escalar [0,1]',
  imagenet: 'ImageNet',
  custom: 'Personalizado',
};

// Valores conocidos de ImageNet (orden RGB).
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

// Campos del input que define cada preset (custom no toca mean/std).
type NormFields = Pick<InputConfig, 'scale' | 'normalize' | 'mean' | 'std'>;

function approx(a: number, b: number): boolean {
  return Math.abs(a - b) < 1e-3;
}

function vecApprox(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => approx(v, b[i]));
}

// mean=0 y std=1 => la normalizacion es identidad (el backend la saltea igual).
function isTrivialMeanStd(mean: number[], std: number[]): boolean {
  return mean.every((v) => approx(v, 0)) && std.every((v) => approx(v, 1));
}

/**
 * Infiere que preset representa un input ya cargado. Sirve para reabrir configs
 * existentes en el wizard sin perder el sentido. Si no matchea ninguno -> 'custom'.
 */
export function inferNormPreset(input: InputConfig): NormPreset {
  const { scale, normalize, mean, std } = input;
  if (!scale && !normalize) return 'none';
  // scale + (sin normalize, o normalize trivial) => solo /255
  if (scale && (!normalize || isTrivialMeanStd(mean, std))) return 'scale01';
  if (scale && normalize && vecApprox(mean, IMAGENET_MEAN) && vecApprox(std, IMAGENET_STD)) {
    return 'imagenet';
  }
  return 'custom';
}

/**
 * Campos a escribir cuando el usuario elige un preset. 'custom' solo garantiza que
 * scale/normalize queden activos y deja mean/std como esten (para que el usuario edite).
 */
export function applyNormPreset(preset: NormPreset, current: InputConfig): NormFields {
  switch (preset) {
    case 'none':
      return { scale: false, normalize: false, mean: [0, 0, 0], std: [1, 1, 1] };
    case 'scale01':
      return { scale: true, normalize: false, mean: [0, 0, 0], std: [1, 1, 1] };
    case 'imagenet':
      return { scale: true, normalize: true, mean: [...IMAGENET_MEAN], std: [...IMAGENET_STD] };
    case 'custom':
      return { scale: true, normalize: true, mean: current.mean, std: current.std };
  }
}

// ── 2. pack_format -> estructura de salida ──────────────────────────────────────

type PackFormat = DetectionOutput['pack_format'];
type CoordsSpace = 'normalized_0_1' | 'tensor_pixels';

interface PackPreset {
  // tensor_structure sin num_classes (num_classes es del modelo, se preserva aparte).
  tensor_structure: Omit<TensorDetection, 'num_classes'>;
  out_coords_space: CoordsSpace;
}

// Constantes por formato. El orden de columnas lo fija el unpacker; las coords mapean
// ese orden al formato estandar [x1,y1,x2,y2,conf,cls] via output_adapter.
//
//   yolo_flat      -> [cx, cy, w, h, score, class_id]
//   tflite_detpost -> [ymin, xmin, ymax, xmax, score, class_id]
//   anchor_deltas  -> [ymin, xmin, ymax, xmax, prob,  class_id]  (verif: efficientdet-lite0/2)
//   boxes_scores   -> ya estandar (no usa adapter; indices solo para validar el schema)
//   raw            -> arbitrario por modelo: NO se autocompleta (lo edita el usuario)
const PACK_PRESETS: Partial<Record<PackFormat, PackPreset>> = {
  yolo_flat: {
    tensor_structure: {
      box_format: 'cxcywh',
      coordinates: { cx: 0, cy: 1, w: 2, h: 3 },
      confidence_index: 4,
      class_index: 5,
    },
    out_coords_space: 'tensor_pixels',
  },
  tflite_detpost: {
    tensor_structure: {
      box_format: 'yxyx',
      coordinates: { y1: 0, x1: 1, y2: 2, x2: 3 },
      confidence_index: 4,
      class_index: 5,
    },
    out_coords_space: 'normalized_0_1',
  },
  anchor_deltas: {
    tensor_structure: {
      box_format: 'yxyx',
      coordinates: { y1: 0, x1: 1, y2: 2, x2: 3 },
      confidence_index: 4,
      class_index: 5,
    },
    out_coords_space: 'tensor_pixels',
  },
  boxes_scores: {
    tensor_structure: {
      box_format: 'xyxy',
      coordinates: { x1: 0, y1: 1, x2: 2, y2: 3 },
      confidence_index: 4,
      class_index: 5,
    },
    out_coords_space: 'normalized_0_1',
  },
};

// anchor_deltas SIEMPRE entrega pixeles del tensor: el espacio queda fijo (no editable).
export function isCoordsSpaceLocked(format: PackFormat): boolean {
  return format === 'anchor_deltas';
}

/**
 * Resultado a aplicar al elegir un pack_format. Para 'raw' no hay autocompletado:
 * devuelve null y el usuario edita la estructura a mano (queda expandida).
 * num_classes se preserva (es del modelo, no del formato).
 */
export function applyPackPreset(
  format: PackFormat,
  currentNumClasses: number | null,
): { tensor_structure: TensorDetection; out_coords_space: CoordsSpace } | null {
  const preset = PACK_PRESETS[format];
  if (!preset) return null;
  return {
    tensor_structure: { ...preset.tensor_structure, num_classes: currentNumClasses },
    out_coords_space: preset.out_coords_space,
  };
}

// ── 3. device -> ejecucion ──────────────────────────────────────────────────────

// gpu intenta CUDA y cae a CPU; cpu solo CPU. (El orden es prioridad para ORT.)
export function providersForDevice(device: 'cpu' | 'gpu'): string[] {
  return device === 'gpu'
    ? ['CUDAExecutionProvider', 'CPUExecutionProvider']
    : ['CPUExecutionProvider'];
}

export function delegatesForDevice(device: 'cpu' | 'gpu'): string[] {
  return device === 'gpu' ? ['gpu'] : [];
}

// Backends que exponen seleccion de providers/delegates derivada de device.
export function backendHasExecutionOpts(backend: Backend): boolean {
  return backend === 'onnxruntime' || backend === 'tflite';
}
