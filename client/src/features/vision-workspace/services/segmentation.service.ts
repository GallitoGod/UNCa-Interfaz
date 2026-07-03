// segmentation.service.ts — estrategia de SEGMENTACION (STUB).
// implemented:false -> el workspace muestra UnsupportedOverlay.
// Llenar cuando exista contrato de segmentacion en el backend (hoy 501).

import type { VisionStrategy, VisionFrameContext } from './types';

// PROPUESTA (no es contrato): mascara por pixel (argmax_map) o probabilidades.
export interface SegmentationResult {
  width: number;
  height: number;
  mask: Uint8Array | number[]; // id de clase por pixel
}

export const segmentationStrategy: VisionStrategy<SegmentationResult> = {
  type: 'segmentation',
  implemented: false,

  // TODO(contrato): leer payload de mascara -> { width, height, mask }
  parse(): SegmentationResult | null {
    return null;
  },

  // TODO(contrato): pintar mascara coloreada (colormap) sobre el canvas con alpha.
  // Recomendado: canvas offscreen para la mascara + drawImage con globalAlpha.
  present(_result: SegmentationResult, _frame: VisionFrameContext) {
    /* pendiente: contrato de segmentacion */
  },

  clear(_frame: VisionFrameContext) {
    // El repintado del frame por el workspace limpia la mascara del canvas.
  },
};
