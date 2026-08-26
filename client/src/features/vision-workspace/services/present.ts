// present.ts — render de UN frame en el workspace.
//
// Dos caminos, segun lo que devolvio el backend (paso 3, 2026-08-26):
//
//   BINARIO ('frame') -> deteccion/segmentacion: el backend YA compuso el frame con
//                        supervision. El workspace lo pinta y no hay estrategia que
//                        invocar: el cliente no dibuja ni una caja.
//   TEXTO   ('json')  -> clasificacion y errores: vale la regla de siempre, el
//                        workspace repinta el frame base y la estrategia agrega su
//                        capa (el panel HTML) encima.

import { getStrategy } from './registry';
import type { DrawSettings, ModelType, VisionFrameContext } from './types';
// Type-only: no genera dependencia en runtime entre features.
import type { StreamPayload } from '@/features/inference/services/videoStream';

interface PresentArgs {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  overlayRoot: HTMLElement;
  source: HTMLCanvasElement; // el frame capturado (intacto) a repintar
  payload: StreamPayload; // respuesta del WS, ya discriminada por el transporte
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

/**
 * Desmonta YA la capa de la estrategia presentada por ultima vez en este overlay.
 *
 * releasePrevious() solo corre cuando llega un frame nuevo. Con una fuente estatica
 * (imagen) no hay frame siguiente: al cambiar de modelo la capa vieja se quedaba
 * pegada en pantalla hasta que el usuario cambiaba de fuente. Esto la suelta en el
 * momento del cambio, sin esperar. El registro de "que habia" sigue viviendo aca,
 * asi que no hay dos fuentes de verdad.
 */
export function releaseOverlay(overlayRoot: HTMLElement, frame: VisionFrameContext): void {
  releasePrevious(overlayRoot, frame, null);
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
  // 1. Frame ya compuesto por el backend: pintarlo tal cual y salir.
  if (payload.kind === 'frame') {
    const bitmap = payload.bitmap;
    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }
    ctx.drawImage(bitmap, 0, 0);
    // Liberar el bitmap YA: a 30 fps son cientos por minuto y el GC no los apura.
    bitmap.close();

    // Aunque no haya capa que agregar, sigue habiendo capa que SACAR: si el modelo
    // anterior era clasificacion, su panel HTML sobrevive al repintado del canvas.
    releasePrevious(
      overlayRoot,
      { canvas, ctx, overlayRoot, frameWidth: canvas.width, frameHeight: canvas.height, settings: drawSettings },
      modelType,
    );
    return;
  }

  // 2. Camino JSON: repintar el frame base (el backend no lo compuso).
  const envelope = payload.envelope;
  if (canvas.width !== source.width || canvas.height !== source.height) {
    canvas.width = source.width;
    canvas.height = source.height;
  }
  ctx.drawImage(source, 0, 0);

  // 3. Contexto del frame (barato: literal). Se arma antes de los early-returns
  //    porque clear() tambien lo necesita.
  const frame: VisionFrameContext = {
    canvas,
    ctx,
    overlayRoot,
    frameWidth: source.width,
    frameHeight: source.height,
    settings: drawSettings,
  };

  // 4. Estrategia activa. Sin tipo o no implementada: no hay capa que presentar
  //    (el frame ya esta dibujado y VisionWorkspace muestra UnsupportedOverlay).
  const strategy = modelType ? getStrategy(modelType) : null;
  const active = strategy && strategy.implemented ? strategy : null;
  releasePrevious(overlayRoot, frame, active ? active.type : null);

  if (!active) return;

  // 5. Errores del backend: no hay resultados, asi que se limpia la capa para no
  //    dejar en pantalla el resultado del ultimo frame bueno.
  const err = (envelope as ErrorPayload | null)?.error;
  if (err) {
    // no_model es un estado normal antes de seleccionar modelo, no se loguea.
    if (err !== 'no_model') {
      console.warn('Error de stream:', err, '- ver /logs/inference');
    }
    active.clear(frame);
    return;
  }

  // 6. Parse + present (try/catch: un fallo de frame no mata el loop).
  try {
    const result = active.parse(envelope);
    if (result !== null) active.present(result, frame);
    else active.clear(frame);
  } catch (e) {
    console.error('Fallo al presentar frame:', e);
  }
}
