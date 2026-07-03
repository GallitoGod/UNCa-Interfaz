// registry.ts — despacho de estrategia por model_type. Unico lugar de despacho:
// agregar un tipo nuevo = crear <tipo>.service.ts + registrarlo aca.

import type { ModelType, VisionStrategy } from './types';
import { detectionStrategy } from './detection.service';
import { classificationStrategy } from './classification.service';
import { segmentationStrategy } from './segmentation.service';

export const VISION_STRATEGIES: Record<ModelType, VisionStrategy> = {
  detection: detectionStrategy,
  classification: classificationStrategy, // stub
  segmentation: segmentationStrategy, // stub
};

export function getStrategy(type: ModelType): VisionStrategy {
  return VISION_STRATEGIES[type];
}
