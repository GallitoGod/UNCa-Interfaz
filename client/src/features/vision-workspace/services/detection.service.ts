// detection.service.ts — estrategia de DETECCION.
//
// ⚠️ Esta estrategia YA NO DIBUJA. Desde el 2026-08-26 (paso 3 del plan del
// 2026-08-21) el backend compone el frame con los annotators de supervision y lo
// manda por el WS como JPEG binario; el workspace solo lo pinta (present.ts, rama
// 'frame'). El dibujo de cajas que vivia aca —heredado del overlay.js viejo— se
// borro: el cliente es un thin client puro y no contiene ni una linea que dibuje
// una caja. Si hace falta ver como era, esta en el historial de git.
//
// Entonces, por que sigue existiendo el archivo:
//   1. el registry necesita una entrada por model_type (registry.ts),
//   2. VisionWorkspace pregunta 'implemented' para decidir si muestra el
//      UnsupportedOverlay — deteccion SI esta soportada,
//   3. present.ts llama clear() sobre la estrategia ANTERIOR al cambiar de tipo;
//      deteccion no deja capa HTML, pero tiene que existir para ser llamada.
//
// Los colores de las cajas ahora viajan al backend por POST /config/draw
// (features/inference/api/drawSettings.ts).

import type { VisionStrategy } from './types';

export const detectionStrategy: VisionStrategy<null> = {
  type: 'detection',
  implemented: true,

  // El backend no manda datos de deteccion por el envelope: manda el frame ya
  // compuesto, que present.ts atiende antes de llegar aca. Si igual llegara un
  // envelope de deteccion (backend viejo), no hay nada que presentar.
  parse() {
    return null;
  },

  present() {
    // Sin cuerpo a proposito: dibujar es del backend.
  },

  clear() {
    // Deteccion no monta capa HTML; el repintado del canvas ya limpia todo.
  },
};
