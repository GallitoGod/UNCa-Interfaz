// present.ts — render de UN frame en el workspace. Regla de oro: el workspace
// repinta el frame base; la estrategia solo agrega su capa encima.

import { getStrategy } from './registry';
import type { DrawSettings, ModelType, VisionFrameContext } from './types';

interface PresentArgs {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  overlayRoot: HTMLElement;
  source: HTMLCanvasElement; // el frame capturado (intacto) a repintar
  payload: unknown; // respuesta cruda del WS
  modelType: ModelType | null;
  drawSettings: DrawSettings;
}

interface ErrorPayload {
  error?: string | null;
}

export function presentFrame({
  canvas,
  ctx,
  overlayRoot,
  source,
  payload,
  modelType,
  drawSettings,
}: PresentArgs): void {
  // 1. Repintar el frame base (comun a todos los tipos).
  if (canvas.width !== source.width || canvas.height !== source.height) {
    canvas.width = source.width;
    canvas.height = source.height;
  }
  ctx.drawImage(source, 0, 0);

  // 2. Errores del backend.
  const err = (payload as ErrorPayload | null)?.error;
  if (err === 'no_model') return; // estado normal antes de seleccionar modelo
  if (err) {
    console.warn('Error de stream:', err, '- ver /logs/inference');
    return;
  }

  // 3. Estrategia segun tipo. Sin tipo o no implementada: no presenta (el frame ya esta).
  if (!modelType) return;
  const strategy = getStrategy(modelType);
  if (!strategy.implemented) return;

  // 4. Parse + present (try/catch: un fallo de frame no mata el loop).
  const frame: VisionFrameContext = {
    canvas,
    ctx,
    overlayRoot,
    frameWidth: source.width,
    frameHeight: source.height,
    settings: drawSettings,
  };
  try {
    const result = strategy.parse(payload);
    if (result !== null) strategy.present(result, frame);
  } catch (e) {
    console.error('Fallo al presentar frame:', e);
  }
}
