// toBackendConfig.ts — transforma el estado del wizard al shape que espera el
// schema estricto del backend, justo antes de guardar. Funcion pura y testeable.
//
// El wizard mantiene out_coords_space dentro de output por comodidad de edicion;
// el schema lo quiere en runtime.runtimeShapes. Ademas anula el backend no usado y
// el anchor_config cuando no aplica.

import type { ModelConfig } from '@/shared/api/types';

// Shape laxo del estado interno del wizard (output puede traer out_coords_space extra).
interface WizardConfig {
  model_type: string;
  output: Record<string, unknown> & { out_coords_space?: string; pack_format?: string };
  runtime: Record<string, unknown> & {
    backend?: string;
    runtimeShapes?: Record<string, unknown> | null;
    onnx?: unknown;
    tflite?: unknown;
  };
  [k: string]: unknown;
}

export function toBackendConfig(state: ModelConfig): ModelConfig {
  const out = structuredClone(state) as unknown as WizardConfig;

  // 1. out_coords_space vive en runtime.runtimeShapes, no en output.
  if (out.output.out_coords_space) {
    out.runtime.runtimeShapes ??= {};
    (out.runtime.runtimeShapes as Record<string, unknown>).out_coords_space =
      out.output.out_coords_space;
    delete out.output.out_coords_space;
  }

  // 2. Anular el backend no utilizado.
  if (out.runtime.backend !== 'tflite') out.runtime.tflite = null;
  if (out.runtime.backend !== 'onnxruntime') out.runtime.onnx = null;

  // 3. anchor_config solo aplica a deteccion con pack_format anchor_deltas.
  if (out.model_type === 'detection' && out.output.pack_format !== 'anchor_deltas') {
    out.output.anchor_config = null;
  }

  return out as unknown as ModelConfig;
}
