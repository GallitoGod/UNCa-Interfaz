# Componente: Vision Workspace

Diseño del **vision-workspace**: la superficie de presentación de resultados de inferencia
en vivo del nuevo frontend React. Es el componente del canvas, rediseñado con una
arquitectura de **estrategias por tipo de modelo** (detection / classification /
segmentation) para que sea mantenible y expansible.

Este doc es el **plano** del diseño; al 2026-06-26 ya está **implementado** (ver banner).

> **✅ Estado (2026-06-26): implementado** en `client/src/features/vision-workspace/`. El
> diseño de abajo se construyó tal cual: `present.ts` (regla de oro frame-base + estrategia,
> manejo de errores del hot path con try/catch por frame, `no_model`→frame solo), `registry.ts`
> (despacho por `model_type`), `detection.service.ts` (implementado: coords px, label dentro de
> caja si toca el borde, fallback a id de clase), `classification`/`segmentation.service.ts`
> (stubs `implemented:false` con checklists), `UnsupportedOverlay` y `workspaceStore`
> (`drawSettings` **persistido en localStorage**). El orquestador del loop vive en
> `useVisionSession` (feature inference), no en un `useVisionWorkspace` propio. El
> `ModelSelector` ya alimenta `setActiveModel(name, type)` con el `model_type` **real**
> (`GET /configs/{name}`), así que el enrutado por tipo y el `UnsupportedOverlay` quedan
> operativos para cuando CLS/SEG sean cargables en el backend (hoy 501 al cargar).

> **Estado de los tipos (al 2026-06-25):** solo **detection** tiene contrato (back y
> front). **classification** y **segmentation** NO tienen contrato ni en el backend
> (`model_controller.py` aún hace `if model_type != "detection": raise NotImplementedError`)
> ni en el frontend. Por eso sus estrategias se dejan como **scaffolding** (`implemented:false`)
> para llenarlas cuando el contrato exista. Las "formas de payload" que aparecen abajo para
> CLS/SEG son **propuestas, NO contrato**.

---

## 1. Responsabilidad y límites

**Responsabilidad:** montar el `<video>`, el canvas de salida y un contenedor de overlays
HTML; por cada mensaje del stream, **repintar el frame base y delegar la presentación del
resultado en la estrategia del tipo de modelo activo**.

Reemplaza la mezcla actual de `overlay.js` (dibujo) + el cableado de canvas disperso en
`scripts.js`.

**Lo que el workspace NO hace** (vecinos en la feature Inferencia, no parte del workspace):

- **Transporte:** la conexión WebSocket y el loop de captura de frames los provee el
  `videoStream` service (ver [`feature-inferencia.md`](./feature-inferencia.md),
  VideoStreamClient). El workspace lo **consume**, no lo implementa.
- **Fuentes:** cámara (`CameraSource`) y archivo (`FileSource`).
- **Controles:** confianza, métricas, logs, modal de colores.

El workspace recibe del exterior: el stream (inyectado) y el `model_type` del modelo activo.

### Regla de oro de composición

**El workspace repinta el frame base; la estrategia solo agrega su capa encima** (nunca
repinta el frame). Así detección (cajas), segmentación (máscara) y clasificación (badge
HTML) componen sin pisarse y se pueden combinar a futuro.

---

## 2. Interfaz común (el contrato de estrategia)

`features/vision-workspace/services/types.ts`

```ts
export type ModelType = 'detection' | 'classification' | 'segmentation';

// Ajustes de dibujo, extensibles por tipo (detección usa colores de caja;
// segmentación sumará colormap/alpha; clasificación, posición/estilo del badge).
export interface DrawSettings {
  bboxColor: string;
  labelColor: string;
  maskAlpha?: number;                 // segmentación (futuro)
  colormap?: string[];               // segmentación (futuro)
}

// Todo lo que una estrategia necesita para presentar un frame.
export interface VisionFrameContext {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;      // capa canvas (cajas / máscaras)
  overlayRoot: HTMLElement;           // capa HTML (badges de clasificación, leyendas)
  frameWidth: number;                 // dimensiones de la imagen original (px)
  frameHeight: number;
  settings: DrawSettings;
  labelMap?: Record<number, string>;  // opcional, a futuro (hoy se muestra el id numérico)
}

// Estrategia por tipo de modelo: parsea el payload crudo del WS y lo presenta.
export interface VisionStrategy<TResult = unknown> {
  readonly type: ModelType;
  readonly implemented: boolean;      // false en los stubs (CLS/SEG)

  // Interpreta el payload crudo del WS para este tipo.
  // Devuelve null si no hay nada que presentar.
  parse(payload: unknown): TResult | null;

  // Presenta el resultado parseado sobre el canvas y/o el overlayRoot.
  present(result: TResult, frame: VisionFrameContext): void;

  // Limpia overlays/estado (al cambiar de modelo o detener el stream).
  clear(frame: VisionFrameContext): void;
}
```

Notas de diseño:

- **`parse` es una función pura** (payload → resultado tipado). Es el punto testeable.
- **`present` es imperativa** sobre canvas/DOM. No re-renderiza React por frame (eso
  mataría el FPS): el `<video>`, el canvas y el `overlayRoot` se montan una vez con `ref`.
- El resultado tipado `TResult` es interno a cada estrategia; el workspace lo trata como
  opaco (solo verifica `!== null`).

---

## 3. Estructura de archivos (destino)

```
features/vision-workspace/
├── components/
│   ├── VisionWorkspace.tsx        # monta video+canvas+overlayRoot; orquesta el loop
│   └── UnsupportedOverlay.tsx     # "este tipo aún no tiene visualización"
├── services/
│   ├── types.ts                   # VisionStrategy, VisionFrameContext, DrawSettings, ModelType
│   ├── registry.ts                # mapa model_type -> strategy
│   ├── detection.service.ts       # IMPLEMENTADO (migra overlay.js)
│   ├── classification.service.ts  # STUB (implemented:false)
│   └── segmentation.service.ts    # STUB (implemented:false)
├── store/
│   └── workspaceStore.ts          # modelo activo {name, type}, drawSettings, fuente activa
└── hooks/
    └── useVisionWorkspace.ts      # conecta el stream y corre parse/present con la estrategia activa
```

### Registry (despacho por tipo)

`features/vision-workspace/services/registry.ts`

```ts
import type { ModelType, VisionStrategy } from './types';
import { detectionStrategy } from './detection.service';
import { classificationStrategy } from './classification.service';
import { segmentationStrategy } from './segmentation.service';

export const VISION_STRATEGIES: Record<ModelType, VisionStrategy> = {
  detection: detectionStrategy,
  classification: classificationStrategy, // stub
  segmentation: segmentationStrategy,     // stub
};

export function getStrategy(type: ModelType): VisionStrategy {
  return VISION_STRATEGIES[type];
}
```

Agregar un tipo nuevo en el futuro = crear `<tipo>.service.ts` + registrarlo acá. Un solo
lugar de despacho.

---

## 4. Flujo de datos (por mensaje del WS)

```
stream.onMessage(payload):
  1. workspace repinta el frame capturado en el canvas            (común a todos)
  2. si payload.error === "no_model"  → frame solo, return        (estado normal)
     si payload.error (cualquier otro) → log a consola, return
  3. strategy = getStrategy(activeModel.type)
  4. si !strategy.implemented → mostrar UnsupportedOverlay, return
  5. result = strategy.parse(payload)
  6. si result !== null → strategy.present(result, frameCtx)
        (envuelto en try/catch: el fallo de un frame no mata el loop)
```

- **Repintado del frame (paso 1):** lo hace el workspace, no la estrategia. Mantiene la
  invariante "1 frame en vuelo + repintar el frame que produjo estas detecciones" del
  VideoStreamClient actual.
- **`activeModel.type` (paso 3):** sale del **config del modelo** al seleccionarlo
  (`uncaAPI.readConfig(baseName).model_type`) y vive en `workspaceStore`. **No se toca el
  contrato del WS**: el tipo se conoce client-side, no viene en la respuesta.
- **Cambio de modelo / fuente:** antes de empezar con la nueva, se llama
  `strategy.clear(frameCtx)` de la anterior para limpiar overlays HTML residuales.

### El store

`workspaceStore` (Zustand) mantiene:

```ts
interface WorkspaceState {
  activeModel: { name: string; type: ModelType } | null;
  drawSettings: DrawSettings;
  setActiveModel(name: string, type: ModelType): void;
  setDrawSettings(patch: Partial<DrawSettings>): void;
}
```

El `model_type` se setea al seleccionar modelo (lo provee la feature Modelos / el
ModelSelector leyendo el config). `drawSettings` lo edita el DrawSettingsModal.

---

## 5. Las tres estrategias

### 5.1 detection.service.ts — IMPLEMENTADO

Migra la lógica de `overlay.js` (hoy en `src/render/modules/overlay.js`).

```ts
import type { VisionStrategy, VisionFrameContext } from './types';

// [x1, y1, x2, y2, conf, cls] en px de la imagen original
type Detection = [number, number, number, number, number, number];

export const detectionStrategy: VisionStrategy<Detection[]> = {
  type: 'detection',
  implemented: true,

  // El WS de detección responde { detections: [[x1,y1,x2,y2,conf,cls], ...], error }
  parse(payload): Detection[] | null {
    const dets = (payload as { detections?: Detection[] })?.detections;
    return dets && dets.length > 0 ? dets : null;
  },

  present(detections, { ctx, settings }: VisionFrameContext) {
    ctx.lineWidth = 2;
    ctx.font = 'bold 14px sans-serif';
    ctx.textBaseline = 'alphabetic';
    for (const [x1, y1, x2, y2, conf, cls] of detections) {
      ctx.strokeStyle = settings.bboxColor;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `${Math.round(cls)} ${conf.toFixed(2)}`;
      const tw = ctx.measureText(label).width;
      const labelH = 18;
      const ly = y1 >= labelH ? y1 : y1 + labelH; // si toca el borde superior, etiqueta adentro

      ctx.fillStyle = settings.bboxColor;
      ctx.fillRect(x1, ly - labelH, tw + 8, labelH);
      ctx.fillStyle = settings.labelColor;
      ctx.fillText(label, x1 + 4, ly - 5);
    }
  },

  clear() {
    // Detección dibuja solo en canvas; el repintado del frame por el workspace ya lo limpia.
  },
};
```

Reglas de negocio (heredadas de `overlay.js`):

- Coordenadas en **px de la imagen original** (el backend ya deshizo el letterbox).
- La etiqueta muestra el **id numérico de clase** salvo que llegue `labelMap` (futuro).
- Si la caja toca el borde superior, la etiqueta se dibuja **dentro** de la caja.

### 5.2 classification.service.ts — STUB

```ts
import type { VisionStrategy, VisionFrameContext } from './types';

// PROPUESTA (no es contrato): el backend de clasificación devolvería algo como
//   { classification: [{ class_id: number, score: number }, ...], error }
export interface ClassificationResult {
  predictions: { classId: number; score: number }[];
}

export const classificationStrategy: VisionStrategy<ClassificationResult> = {
  type: 'classification',
  implemented: false, // ← scaffolding: el workspace muestra UnsupportedOverlay

  // TODO(contrato): leer payload.classification -> { predictions }
  parse() {
    return null;
  },

  // TODO(contrato): renderizar un badge HTML en overlayRoot con top-k clases + score.
  // Clasificación NO se dibuja en el canvas (es texto): usar overlayRoot, no ctx.
  present(_result, _frame: VisionFrameContext) {
    /* pendiente: contrato de clasificación */
  },

  clear(_frame: VisionFrameContext) {
    // TODO: remover el badge del overlayRoot.
  },
};
```

Checklist para implementarla cuando exista el contrato:

1. Definir el payload real del WS para clasificación (acordar con el backend).
2. `parse`: mapear payload → `ClassificationResult` (aplicar top-k client-side si hace
   falta, o confiar en el backend).
3. `present`: crear/actualizar un nodo HTML en `overlayRoot` (badge con clase + score).
   Considerar `labelMap` para mostrar nombres en vez de ids.
4. `clear`: remover el badge.
5. Flip `implemented: true`.

### 5.3 segmentation.service.ts — STUB

```ts
import type { VisionStrategy, VisionFrameContext } from './types';

// PROPUESTA (no es contrato): máscara por píxel (argmax_map) o probabilidades.
export interface SegmentationResult {
  width: number;
  height: number;
  mask: Uint8Array | number[]; // id de clase por píxel (argmax_map)
}

export const segmentationStrategy: VisionStrategy<SegmentationResult> = {
  type: 'segmentation',
  implemented: false, // ← scaffolding

  // TODO(contrato): leer payload de máscara -> { width, height, mask }
  parse() {
    return null;
  },

  // TODO(contrato): pintar una máscara coloreada (colormap) sobre el canvas con alpha.
  // Recomendado: canvas offscreen para la máscara + drawImage con globalAlpha.
  present(_result, _frame: VisionFrameContext) {
    /* pendiente: contrato de segmentación */
  },

  clear(_frame: VisionFrameContext) {
    // El repintado del frame por el workspace limpia la máscara del canvas.
  },
};
```

Checklist para implementarla cuando exista el contrato:

1. Definir cómo viaja la máscara por el WS (forma + codificación; una máscara por píxel es
   pesada → evaluar RLE/PNG/base64 vs ndarray plano).
2. `parse`: payload → `SegmentationResult`.
3. `present`: construir un `ImageData`/canvas offscreen coloreando por `settings.colormap`,
   y `drawImage` sobre el frame con `settings.maskAlpha`.
4. Considerar `output_stride`/`resize_to_input` (redimensionar la máscara al frame).
5. Flip `implemented: true`.

---

## 6. Errores, estados y testing

### Manejo de errores

- **Tipo no soportado** (`implemented:false`): el workspace muestra `UnsupportedOverlay`
  ("Clasificación: visualización aún no implementada") y **no** intenta parsear/presentar.
  Sin throw en el hot path. Consistente con el 501 honesto del backend.
- **`payload.error`**: `"no_model"` → frame solo (normal); otro valor → log a consola
  (los detalles están en `/logs/inference`).
- **Fallo en `present`**: envuelto en try/catch por frame; un error puntual no rompe el
  loop ni la conexión.

### Testing

| Qué | Cómo |
|---|---|
| `detection.parse` | Función pura: payloads de ejemplo → `Detection[] | null`. Hay samples reales de detección. |
| `detection.present` | Canvas mock (jsdom/OffscreenCanvas): verificar que se llaman `strokeRect`/`fillText`. Opcional. |
| Stubs CLS/SEG | Afirmar `implemented === false`, `parse() === null`, y que el workspace muestra `UnsupportedOverlay` al seleccionarlos. |
| Despacho | `getStrategy(type)` devuelve la estrategia correcta por tipo. |

---

## 7. Resumen de decisiones (del brainstorming)

- **Patrón:** estrategias como **objetos planos** que implementan `VisionStrategy`
  (no clases, no factories).
- **Alcance de cada service:** **Parse + Present** — cada uno interpreta el payload crudo
  de su tipo y decide su superficie (canvas o overlay HTML).
- **Enrutado:** por `model_type` conocido client-side. **No se modifica el contrato del
  WS**; detección lee `payload.detections`.
- **CLS/SEG:** scaffolding con `implemented:false`; payloads documentados como **propuesta,
  no contrato**.
- **Performance:** canvas/DOM imperativos vía `ref`; nada de re-render de React por frame.
