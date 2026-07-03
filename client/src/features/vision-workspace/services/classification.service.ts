// classification.service.ts — estrategia de CLASIFICACION (STUB).
// implemented:false -> el workspace muestra UnsupportedOverlay y no parsea/presenta.
// Llenar cuando exista contrato de clasificacion en el backend (hoy 501).

import type { VisionStrategy, VisionFrameContext } from './types';

// PROPUESTA (no es contrato): el backend devolveria algo como
//   { classification: [{ class_id, score }, ...], error }
export interface ClassificationResult {
  predictions: { classId: number; score: number }[];
}

export const classificationStrategy: VisionStrategy<ClassificationResult> = {
  type: 'classification',
  implemented: false,

  // TODO(contrato): leer payload.classification -> { predictions }
  parse(): ClassificationResult | null {
    return null;
  },

  // TODO(contrato): renderizar un badge HTML en overlayRoot con top-k clases + score.
  // Clasificacion NO se dibuja en el canvas (es texto): usar overlayRoot, no ctx.
  present(_result: ClassificationResult, _frame: VisionFrameContext) {
    /* pendiente: contrato de clasificacion */
  },

  clear(_frame: VisionFrameContext) {
    // TODO: remover el badge del overlayRoot.
  },
};
