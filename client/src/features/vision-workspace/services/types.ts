// types.ts — contrato de las estrategias de presentacion por tipo de modelo.
// Ver docs/frontend-components/vision-workspace.md.

import type { ModelType } from '@/shared/api/types';

export type { ModelType };

// Estilo de marca de una deteccion. Cuatro de la familia de doce que trae
// supervision: el resto seria ruido en la UI (ver render/draw_config.py).
export type BoxStyle = 'box' | 'round' | 'corner' | 'dot';

// Cuanto texto lleva cada deteccion (LABEL_MODES en render/draw_config.py). Existe
// porque con modelos de muchas detecciones los carteles tapan la escena: con 'best'
// sobre material aereo salen ~70 cajas y no se ve ni la imagen ni las cajas.
export type LabelMode = 'completa' | 'corta' | 'ninguna';

// Ajustes de dibujo. Desde el 2026-08-26 el que dibuja es el BACKEND: esto es el
// estado del que el cliente es DUENO (lo persiste en localStorage) y que empuja por
// POST /config/draw. El cliente ya no los usa para dibujar nada.
export interface DrawSettings {
  bboxColor: string;
  labelColor: string;
  maskAlpha: number; // segmentacion (futuro)
  boxStyle: BoxStyle;
  labelMode: LabelMode; // cuanto texto lleva cada deteccion (o ninguno)
  smartLabels: boolean; // correr los carteles para que no se tapen entre si. Inerte con labelMode 'ninguna'
  shading: boolean; // rellenar la caja con el color de acento translucido
  autoScale: boolean; // grosor y texto derivados de la resolucion del frame

  // ── Seguimiento (Tier B, 2026-08-27) ──────────────────────────────────────
  // Estos tres son ajustes del USUARIO igual que los de arriba (persisten, viajan
  // por el mismo POST /config/draw), pero habilitan MEMORIA en el backend, que vive
  // por conexion del WebSocket y muere con ella. El cliente no guarda esa memoria:
  // solo dice si la quiere.
  //
  // Solo tienen efecto con camara o video: una imagen fija abre un WS declarado
  // `?stateful=false` y el backend no rastrea nada sobre una foto suelta.
  tracking: boolean; // identidad estable por objeto entre frames (#id en la etiqueta)
  smoothing: boolean; // promedia la posicion en las ultimas N deteccciones. REQUIERE tracking
  smoothingLength: number; // ventana del promedio
  traces: boolean; // estela del recorrido de cada objeto. REQUIERE tracking
  tracesLength: number; // cuantos frames de recorrido conserva la estela
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
