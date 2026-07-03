# Feature: Inferencia

La vista principal: video en vivo (cámara o archivo) con detecciones superpuestas, más
los controles asociados (confianza, métricas, logs, colores de dibujo, grabación).

El contrato central del frontend vive acá: **se envía un frame JPEG binario por
WebSocket y el backend responde UN JSON `{detections, error}` por frame**; el dibujo es
client-side.

Archivos as-is: `streamHandler.js`, `cameraSwitcher.js`, `fileHandler.js`, `overlay.js`,
`record.js`, `modelLoader.js`, `selectModel.js`, y la parte de controles de `scripts.js`.

> **✅ Estado (2026-06-26): implementado** en `client/src/features/inference/` +
> `features/vision-workspace/`. Notas de lo realmente construido (deltas con el to-be de abajo):
> - **Transporte:** `videoStream.ts` conserva las 5 invariantes y suma `pause()`/`resume()`
>   para la pausa al navegar (ver `app-shell.md`). El contrato WS **se mantiene en píxeles**
>   `{detections, error}` (el Anexo A.3 del SDD propone un sobre nuevo normalizado 0–1 → se
>   dejó como visión futura; el back ya entrega px y deshace el letterbox).
> - **ModelSelector:** ya **lee el `model_type` real** del config (`GET /configs/{name}`, no
>   IPC) y hace `setActiveModel(name, type)`; con fallback a `detection` si no se puede leer.
> - **Recorder:** `useRecorder` + librería **`fix-webm-duration`** (inyecta la duración antes
>   de descargar; regla estricta del SDD). Fallback al blob crudo si la lib falla.
> - **ConfidenceSlider:** debounce (200 ms). **DrawSettings:** persistido en `localStorage`.
> - **MetricsHUD/LogPanel:** `useQuery` con `refetchInterval` + `enabled` atado a abierto.
> - Los hooks de diagnóstico viven en `useDiagnostics.ts` (no en archivos sueltos por hook).

---

## VideoStreamClient

- **Responsabilidad:** Gestiona la conexión WebSocket de inferencia: captura frames de un
  `<video>`, los envía como JPEG binario y, al recibir la respuesta, repinta el frame y
  delega el dibujo de cajas. Maneja reconexión y anti-deadlock.
- **Entradas:**
  - `videoElement` y opción `{ mirror }` (lo pasa CameraSource o FileSource).
  - Frames del `<video>` vía un canvas de captura oculto.
  - Mensajes WS entrantes: JSON `{ detections: [[x1,y1,x2,y2,conf,cls], ...], error }`.
- **Salidas:**
  - Frames JPEG binarios al backend (`ws.send(blob)`, calidad 0.8).
  - Dibujo en `#outputCanvas` (repinta el frame capturado + cajas vía DetectionOverlay).
  - Devuelve un handle `{ close() }`.
- **Dependencias:** `constants.streamUrl`, `overlay.drawDetections`, `#outputCanvas`,
  APIs del browser (`WebSocket`, `requestAnimationFrame`, `canvas.toBlob`,
  `performance.now`).
- **Reglas de negocio (críticas):**
  - **Un solo frame en vuelo:** no envía el siguiente hasta recibir respuesta
    (`waitingForResponse`). Evita saturar el backend.
  - **Anti-deadlock:** si no llega respuesta en `RESPONSE_TIMEOUT_MS = 3000`, libera la
    espera y sigue. Red de seguridad si el backend dejara de responder un frame.
  - **El frame enviado sigue intacto en el canvas de captura** hasta la próxima captura;
    por eso se puede repintar exactamente el frame que produjo esas detecciones.
  - **`error: "no_model"` es estado normal** (antes de seleccionar modelo): muestra el
    frame sin cajas, no es un fallo.
  - **Reconexión con backoff exponencial** (1s → máx 10s); no reconecta si el cierre fue
    intencional (`close()`).
  - El espejo (`mirror`) se aplica acá en la captura, **solo cuando lo pide CameraSource**.
- **Mapeo al destino React:** `features/inference/services/videoStream.ts` (clase/closure
  framework-agnóstica, **no** un hook que re-renderiza). El estado observable (conectado,
  esperando, último error) va a un **store Zustand**. **No usar TanStack Query.** El
  `<video>`/canvas se montan en un componente con `ref`; el loop sigue sobre
  `requestAnimationFrame`, fuera del ciclo de render de React.

## CameraSource

- **Responsabilidad:** Enumerar cámaras, iniciar la seleccionada vía `getUserMedia` y
  abrir el stream de inferencia con espejo. Maneja cambio y refresco de cámaras.
- **Entradas:** elemento `<select>` de cámaras; eventos `change`/refresh; permisos y
  dispositivos de `navigator.mediaDevices`.
- **Salidas:** asigna `srcObject` al `<video>`; abre un `VideoStreamClient` con
  `{ mirror: true }`; oculta/muestra el mensaje "no hay video".
- **Dependencias:** `streamHandler.initVideoStream`, `navigator.mediaDevices`
  (`getUserMedia`, `enumerateDevices`).
- **Reglas de negocio:**
  - **El espejo es client-side y SOLO para cámara** (`mirror: true`). Los archivos nunca
    se espejan.
  - Al cambiar de cámara, cierra el WS anterior y detiene los tracks previos antes de
    abrir el nuevo (evita fugas de stream).
  - Pide permiso con `getUserMedia` antes de poblar la lista (si no, los `label` vienen
    vacíos) y, tras refrescar, intenta **mantener** la cámara previamente seleccionada.
  - Expone `stopCurrentStream()` para que FileSource libere la cámara al subir un archivo.
- **Mapeo al destino React:** `features/inference/components/CameraSource` + lógica en
  `store/streamStore` (fuente activa, deviceId, mirror). La lista de dispositivos puede
  ser un hook propio (no Query: es API del browser, no del backend).

## FileSource

- **Responsabilidad:** Cargar un archivo local de **video** o **imagen** y enviarlo a
  inferencia (sin espejo).
- **Entradas:** un `File` (de un `<input type=file>`); su `file.type`.
- **Salidas:**
  - Video → `video.src = objectURL`, loop+muted, abre `VideoStreamClient { mirror:false }`.
  - Imagen → abre un WS efímero, envía **un** frame (calidad 0.9), dibuja el resultado y
    cierra.
- **Dependencias:** `streamHandler.initVideoStream`, `cameraSwitcher.stopCurrentStream`,
  `constants.streamUrl`, `overlay.drawDetections`, `#outputCanvas`.
- **Reglas de negocio:**
  - **Sin espejo** (los archivos van tal cual).
  - Antes de iniciar, **detiene la cámara** (`stopCurrentStream`) y cierra cualquier WS de
    archivo previo.
  - El **video** reusa el VideoStreamClient (loop continuo); la **imagen** usa un WS
    one-shot dedicado (un frame, dibuja, cierra) — no pasa por el loop de frames.
- **Mapeo al destino React:** `features/inference/components/FileSource`. El caso imagen
  (WS one-shot) se vuelve una función del `videoStream` service; el caso video reusa el
  mismo service que la cámara con `mirror:false`. Estado de fuente en `streamStore`.

## DetectionOverlay (+ drawSettings)

- **Responsabilidad:** Dibujar las cajas y etiquetas sobre el canvas de salida. Guarda los
  colores de dibujo (`drawSettings`) que edita DrawSettingsModal.
- **Entradas:** `ctx` (2D context) y `detections: [[x1,y1,x2,y2,conf,cls], ...]`;
  `drawSettings.bboxColor/labelColor`.
- **Salidas:** strokes/fills en el canvas (cajas + label `"<cls> <conf>"`).
- **Dependencias:** ninguna externa (Canvas 2D API).
- **Reglas de negocio:**
  - Coordenadas en **píxeles de la imagen original** (el backend ya deshace el letterbox).
  - La etiqueta muestra el **id numérico de clase** (no hay `label_map` en el contrato
    actual — ver nota abierta en CLAUDE.md).
  - Si la caja toca el borde superior, la etiqueta se dibuja **dentro** de la caja.
  - El color por defecto del label es `#000000` en el código, pero el `<input>` del modal
    arranca en `#FFFFFF` y Bootstrap lo sincroniza al cargar (la fuente efectiva es el
    input). Detalle a unificar en la migración.
- **Mapeo al destino React:** `features/inference/services/overlay.ts` (función pura de
  dibujo, imperativa). `drawSettings` pasa a un **store Zustand** (`drawSettingsStore`).
  **No** convertir en componente que re-renderice por frame.

## Recorder

- **Responsabilidad:** Grabar el canvas de salida a `.webm` y descargarlo.
- **Entradas:** el botón de grabar y el `#outputCanvas`.
- **Salidas:** archivo `grabacion.webm` (descarga vía `<a download>`); muta el texto/color
  del botón; exporta la flag `isRecording`.
- **Dependencias:** `canvas.captureStream`, `MediaRecorder`, `URL.createObjectURL`.
- **Reglas de negocio:**
  - Graba a 30 fps en `video/webm`.
  - Estado de grabación expuesto como **flag exportada** (`isRecording`) para evitar
    inferir el estado leyendo el texto del botón.
- **Mapeo al destino React:** `features/inference/components/Recorder` + `useRecorder`
  hook; `isRecording` como estado local o en `streamStore`. El estilo del botón pasa a
  clases Tailwind condicionales (hoy se setea con `style.backgroundColor` inline).

## ModelSelector

- **Responsabilidad:** Poblar el `<select>` de modelos disponibles y pedir al backend que
  cargue el elegido.
- **Entradas:** `GET /get_models` (`modelLoader`); `change` del `<select>`; nombre de
  modelo (`selectModel`).
- **Salidas:** `POST /select_model {model_name}`; opciones del `<select>`; log en consola.
- **Dependencias:** `constants.loadModelUrl/selectModelUrl`, `fetch`.
- **Reglas de negocio:**
  - Al cargar, **auto-selecciona el primer modelo** para que el backend tenga uno listo.
  - `selectModel('')` (vacío) es no-op.
  - `/get_models` solo lista configs **con archivo de pesos** (filtro del backend).
- **Mapeo al destino React:** `features/inference/api/models.ts` + hooks
  `useModels` (query) y `useSelectModel` (mutation) de **TanStack Query**. El `<select>`
  pasa a `shared/ui`. La auto-selección del primer modelo se hace en el componente al
  resolver la query.

## ConfidenceControl

- **Responsabilidad:** Slider de umbral de confianza "en vivo".
- **Entradas:** eventos `input` (preview del %) y `change` (commit) del slider.
- **Salidas:** `POST /config/confidence {value: 0..1}`; actualiza el `<span>` del %.
- **Dependencias:** `constants.confidenceUrl`, `fetch`.
- **Reglas de negocio:**
  - El slider está en `[0,100]`; se divide por 100 antes de enviar (`[0,1]`).
  - Solo hace POST en `change` (al soltar), no en cada `input`, para no inundar el backend.
  - El backend valida rango y exige modelo cargado (409 sin modelo, 422 fuera de rango).
- **Mapeo al destino React:** `features/inference/components/ConfidenceSlider` +
  `useUpdateConfidence` (mutation TanStack Query, con debounce). Valor del slider en
  estado local.

## MetricsHUD

- **Responsabilidad:** Mostrar métricas de rendimiento (FPS, inferencia, total, P95) en un
  overlay sobre el video, con polling mientras está abierto.
- **Entradas:** `GET /metrics`; toggle del botón.
- **Salidas:** texto en los `<span>` del HUD; arranca/limpia un `setInterval` (1s).
- **Dependencias:** `constants.metricsUrl`, `fetch`.
- **Reglas de negocio:**
  - Polling **solo mientras el HUD está abierto** (se limpia el intervalo al cerrar).
  - Si `status !== 'ok'` o no hay métricas, muestra `--`.
- **Mapeo al destino React:** `features/inference/components/MetricsHUD` + `useMetrics`
  (TanStack Query con `refetchInterval: 1000`, `enabled` atado a si el HUD está abierto).
  Reemplaza el `setInterval` manual.

## InferenceLogPanel

- **Responsabilidad:** Listar los últimos errores de inferencia, con polling mientras está
  abierto.
- **Entradas:** `GET /logs/inference` (`{ logs: [{ timestamp, error }] }`); toggle.
- **Salidas:** lista `<li>` renderizada (orden inverso: más reciente primero); intervalo
  de 5s.
- **Dependencias:** `constants.inferenceLogsUrl`, `fetch`.
- **Reglas de negocio:**
  - Polling **solo con el panel abierto** (intervalo 5s, se limpia al cerrar).
  - Lista vacía → mensaje "Sin errores registrados.".
  - El backend mantiene los últimos 50 errores en memoria.
- **Mapeo al destino React:** `features/inference/components/LogPanel` +
  `useInferenceLogs` (TanStack Query, `refetchInterval: 5000`, `enabled` atado a abierto).
  El render del HTML por `innerHTML` pasa a JSX (elimina el riesgo de inyección).

## DrawSettingsModal

- **Responsabilidad:** Modal de "Configuración Avanzada" para elegir los colores de caja y
  etiqueta.
- **Entradas:** clicks de abrir/cerrar/guardar; `input` de los `<input type=color>`.
- **Salidas:** escribe `drawSettings.bboxColor/labelColor` (efecto **client-side**, sin
  backend); preview de color en vivo.
- **Dependencias:** `overlay.drawSettings`.
- **Reglas de negocio:**
  - Los colores **no se persisten** ni viajan al backend (el endpoint `/config/colors` fue
    eliminado; el dibujo es client-side).
  - El preview se actualiza en `input`, pero los valores se **commitean a `drawSettings`
    solo al Guardar**.
  - Cierra al hacer click fuera del contenido o en la X.
- **Mapeo al destino React:** `features/inference/components/DrawSettingsModal` sobre
  `shared/ui/Modal`. Escribe en `drawSettingsStore` (Zustand). Mejora sugerida: persistir
  los colores en `localStorage`.

---

## Contrato del WebSocket (resumen, lo comparten VideoStreamClient y FileSource)

- **Out (cliente → backend):** frame JPEG binario (`Blob`).
- **In (backend → cliente):** `{ detections: [[x1,y1,x2,y2,conf,cls], ...], error }`.
  `detections` en píxeles de la imagen original.
- **`error`** posibles: `null`, `"no_model"` (normal), `"frame_invalido"`,
  `"inference_error"`. El WS **siempre responde** un JSON por frame.
- Invariante del cliente: **1 frame en vuelo** + **timeout 3s** anti-deadlock.

---

## Diseño to-be (React)

La **presentación en canvas no se documenta acá**: vive en
[`vision-workspace.md`](./vision-workspace.md) (estrategias por tipo). Esta sección cubre
el **transporte** (WebSocket), las **fuentes** (cámara/archivo) y los **controles**.
Profundidad escalada: profundo para el stream service; medio para fuentes/selector;
liviano para los controles (son hooks de Query/mutation).

### Estructura

```
features/inference/
├── InferenceView.tsx              # compone fuentes + VisionWorkspace + controles
├── services/
│   └── videoStream.ts            # transporte WS (framework-agnóstico)
├── store/
│   └── streamStore.ts            # estado del stream (conectado, esperando, último error)
├── components/
│   ├── CameraSource.tsx
│   ├── FileSource.tsx
│   ├── ModelSelector.tsx
│   ├── ConfidenceSlider.tsx
│   ├── MetricsHUD.tsx
│   ├── LogPanel.tsx
│   ├── Recorder.tsx
│   └── DrawSettingsModal.tsx
├── hooks/
│   ├── useMetrics.ts
│   ├── useInferenceLogs.ts
│   ├── useUpdateConfidence.ts
│   └── useModels.ts              # get_models + select_model
└── api/
    ├── models.ts                 # GET /get_models, POST /select_model
    └── confidence.ts            # POST /config/confidence
```

### VideoStreamClient → `services/videoStream.ts` *(profundo)*

Transporte WS **framework-agnóstico** (no es un hook; no re-renderiza). Captura frames,
los envía como JPEG binario y emite los payloads crudos. El consumidor (VisionWorkspace)
hace `parse`/`present`. Conserva **todas** las invariantes del `streamHandler.js` actual.

```ts
export interface VideoStreamHandle {
  close(): void;
}

export interface VideoStreamOptions {
  videoElement: HTMLVideoElement;
  mirror?: boolean;                       // true solo para cámara
  onMessage: (payload: unknown, captureCanvas: HTMLCanvasElement) => void;
  onStatus?: (s: StreamStatus) => void;   // alimenta streamStore
}

export type StreamStatus = 'connecting' | 'open' | 'closed' | 'waiting';

export function startVideoStream(opts: VideoStreamOptions): VideoStreamHandle;
```

Invariantes que la implementación DEBE preservar (regresión si se rompen):

- **1 frame en vuelo:** no se envía el próximo hasta recibir respuesta (`waitingForResponse`).
- **Anti-deadlock:** si no llega respuesta en `RESPONSE_TIMEOUT_MS = 3000`, libera la espera.
- **El frame enviado queda intacto en el `captureCanvas`** hasta la próxima captura: por eso
  `onMessage` recibe ese canvas para que el workspace repinte exactamente el frame que
  produjo el resultado.
- **Reconexión con backoff exponencial** (1s → máx 10s); no reconecta si el cierre fue
  intencional (`close()`).
- **`mirror`** se aplica en la captura (scale(-1,1)), solo cuando lo pide la cámara.
- Envío binario (`ws.send(blob)`, JPEG calidad 0.8); nada de base64.

El **estado observable** (status, último error) va a `streamStore` (Zustand) vía `onStatus`.
El loop sigue sobre `requestAnimationFrame`, fuera de React. **No usar TanStack Query** (es
un stream long-lived, no request/response).

### CameraSource → `components/CameraSource.tsx` + hook *(medio)*

Enumera cámaras (`enumerateDevices`), arranca la elegida (`getUserMedia`) y abre el stream
con `mirror:true`. Reglas heredadas:

- **Espejo solo cámara.** Al cambiar de cámara, cerrar el stream anterior y detener los
  tracks previos antes de abrir el nuevo (evita fugas).
- Pedir permiso con `getUserMedia` **antes** de poblar la lista (si no, los `label` vienen
  vacíos); tras refrescar, intentar mantener la cámara previa.
- Exponer un `stopCurrentStream()` (en el hook/store) para que FileSource libere la cámara.

La lista de dispositivos es un hook propio (`useCameras`), **no** TanStack Query (es API del
browser). Device activo + `mirror` en `streamStore`.

### FileSource → `components/FileSource.tsx` *(medio)*

- **Video:** `video.src = URL.createObjectURL(file)`, loop+muted, reusa `startVideoStream`
  con `mirror:false`.
- **Imagen:** WS **one-shot** dedicado (un frame calidad 0.9, dibuja, cierra) — no pasa por
  el loop. Se factoriza como `sendSingleFrame(blob)` en `videoStream.ts`.
- Antes de iniciar: `stopCurrentStream()` (libera la cámara) y cerrar cualquier WS de
  archivo previo. **Sin espejo.**

### ModelSelector → `components/ModelSelector.tsx` + `useModels` *(medio)*

```ts
// hooks/useModels.ts
export function useModels() {
  return useQuery({ queryKey: ['models'], queryFn: getModels }); // GET /get_models
}
export function useSelectModel() {
  return useMutation({ mutationFn: selectModel }); // POST /select_model
}
```

Reglas:

- Al resolver la query, **auto-seleccionar el primer modelo** (efecto en el componente).
- **`/get_models` solo lista configs con pesos** (filtro del backend).
- **Importante (cross-feature) — implementado:** al seleccionar un modelo, además de
  `POST /select_model`, el selector lee el `model_type` del config y lo setea en el
  `workspaceStore` (`setActiveModel(name, type)`), porque el vision-workspace enruta por ese
  tipo. El tipo sale de **`GET /configs/{name}`** (`getModelType()` en `api/models.ts`), no de
  IPC. Fallback a `detection` si no se puede leer.

### Controles → hooks de Query/mutation *(liviano)*

| Componente | Hook | Implementación |
|---|---|---|
| **ConfidenceSlider** | `useUpdateConfidence` | `useMutation` → `POST /config/confidence`. Slider `[0,100]` → `/100`. **Debounce** en lugar del `change`-only actual; valor local en el componente. |
| **MetricsHUD** | `useMetrics` | `useQuery` con `refetchInterval: 1000`, `enabled` atado a si el HUD está abierto. Reemplaza el `setInterval`. `--` si `status !== 'ok'`. |
| **LogPanel** | `useInferenceLogs` | `useQuery` con `refetchInterval: 5000`, `enabled` atado a abierto. Render en JSX (elimina el `innerHTML` actual). Orden inverso (más reciente primero). |
| **Recorder** | `useRecorder` | `canvas.captureStream(30)` + `MediaRecorder` → **`fix-webm-duration`** (inyecta duración) → descarga `.webm`. Fallback al blob crudo si la lib falla. `error`/`recording` como estado local. |
| **DrawSettingsModal** | — | `shared/ui/Modal`; escribe en `workspaceStore.drawSettings`. Commit de colores solo al Guardar. **Persistido en `localStorage`** (clave `uncalens-draw-settings`). |
