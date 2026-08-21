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

// Tipo presentado en el frame anterior, por overlayRoot. Repintar el canvas borra
// solo la capa canvas; la capa HTML (badges de clasificacion) sobrevive sola, asi
// que al cambiar de modelo hay que desmontarla explicitamente con strategy.clear().
const lastTypeByOverlay = new WeakMap<HTMLElement, ModelType>();

// Desmonta la capa de la estrategia anterior si la activa cambio.
function releasePrevious(
  overlayRoot: HTMLElement,
  frame: VisionFrameContext,
  currentType: ModelType | null,
): void {
  const prev = lastTypeByOverlay.get(overlayRoot);
  if (prev && prev !== currentType) {
    try {
      getStrategy(prev).clear(frame);
    } catch (e) {
      console.error('Fallo al limpiar la capa de la estrategia anterior:', e);
    }
  }
  if (currentType) lastTypeByOverlay.set(overlayRoot, currentType);
  else lastTypeByOverlay.delete(overlayRoot);
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

  // 2. Contexto del frame (barato: literal). Se arma antes de los early-returns
  //    porque clear() tambien lo necesita.
  const frame: VisionFrameContext = {
    canvas,
    ctx,
    overlayRoot,
    frameWidth: source.width,
    frameHeight: source.height,
    settings: drawSettings,
  };

  // 3. Estrategia activa. Sin tipo o no implementada: no hay capa que presentar
  //    (el frame ya esta dibujado y VisionWorkspace muestra UnsupportedOverlay).
  const strategy = modelType ? getStrategy(modelType) : null;
  const active = strategy && strategy.implemented ? strategy : null;
  releasePrevious(overlayRoot, frame, active ? active.type : null);

  if (!active) return;

  // 4. Errores del backend: no hay resultados, asi que se limpia la capa para no
  //    dejar en pantalla el resultado del ultimo frame bueno.
  const err = (payload as ErrorPayload | null)?.error;
  if (err) {
    // no_model es un estado normal antes de seleccionar modelo, no se loguea.
    if (err !== 'no_model') {
      console.warn('Error de stream:', err, '- ver /logs/inference');
    }
    active.clear(frame);
    return;
  }

  // 5. Parse + present (try/catch: un fallo de frame no mata el loop).
  try {
    const result = active.parse(payload);
    if (result !== null) active.present(result, frame);
    else active.clear(frame);
  } catch (e) {
    console.error('Fallo al presentar frame:', e);
  }
}
