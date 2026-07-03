// types.ts — contrato de las estrategias de presentacion por tipo de modelo.
// Ver docs/frontend-components/vision-workspace.md.

import type { ModelType } from '@/shared/api/types';

export type { ModelType };

// Ajustes de dibujo, extensibles por tipo. Detection usa colores de caja;
// segmentation sumara colormap/alpha; classification, estilo del badge.
export interface DrawSettings {
  bboxColor: string;
  labelColor: string;
  maskAlpha: number; // segmentacion (futuro)
}

// Todo lo que una estrategia necesita para presentar un frame.
export interface VisionFrameContext {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D; // capa canvas (cajas / mascaras)
  overlayRoot: HTMLElement; // capa HTML (badges de clasificacion, leyendas)
  frameWidth: number; // dimensiones de la imagen original (px)
  frameHeight: number;
  settings: DrawSettings;
  labelMap?: Record<number, string>; // opcional, a futuro
}

// Estrategia por tipo: parsea el payload crudo del WS y lo presenta.
export interface VisionStrategy<TResult = unknown> {
  readonly type: ModelType;
  readonly implemented: boolean; // false en los stubs (CLS/SEG)
  parse(payload: unknown): TResult | null; // null = nada que presentar
  present(result: TResult, frame: VisionFrameContext): void;
  clear(frame: VisionFrameContext): void; // limpia overlays/estado
}
