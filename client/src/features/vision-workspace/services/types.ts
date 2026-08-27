// types.ts — contrato de las estrategias de presentacion por tipo de modelo.
// Ver docs/frontend-components/vision-workspace.md.

import type { ModelType } from '@/shared/api/types';

export type { ModelType };

// Estilo de marca de una deteccion. Cuatro de la familia de doce que trae
// supervision: el resto seria ruido en la UI (ver render/draw_config.py).
export type BoxStyle = 'box' | 'round' | 'corner' | 'dot';

// Ajustes de dibujo. Desde el 2026-08-26 el que dibuja es el BACKEND: esto es el
// estado del que el cliente es DUENO (lo persiste en localStorage) y que empuja por
// POST /config/draw. El cliente ya no los usa para dibujar nada.
export interface DrawSettings {
  bboxColor: string;
  labelColor: string;
  maskAlpha: number; // segmentacion (futuro)
  boxStyle: BoxStyle;
  smartLabels: boolean; // correr los carteles para que no se tapen entre si
  shading: boolean; // rellenar la caja con el color de acento translucido
  autoScale: boolean; // grosor y texto derivados de la resolucion del frame
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
