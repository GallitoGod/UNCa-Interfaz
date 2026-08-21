# Paso 3: render en el backend con supervision (el cliente se vuelve thin client puro)

- **Fecha**: 2026-08-21
- **Rama**: `refactor-frontend-react` (crear `render-backend` a partir de ella)
- **Estado**: propuesto — **bloqueado por los pasos 1 y 2** (ver §1.2)
- **Reemplaza**: el "paso 3" de una línea que figuraba en
  `2026-08-13-migracion-python-312-cuda12-design.md` §1

## 1. Contexto y decisión

### 1.1 Qué se decidió, y con qué evidencia

El 2026-08-21 el usuario decidió **mudar el dibujo al backend**. El cliente deja de
renderizar cajas y máscaras y queda como *thin client puro*: saca la foto, la manda, y
muestra lo que vuelve.

La decisión revierte deliberadamente la **Reforma 3** (2026-06-11), que había movido el
dibujo al cliente para evitar la doble compresión JPEG. La objeción se planteó y quedó
**saldada con datos**: el usuario midió el costo con detectores y no encontró cambios
notorios de latencia contra el esquema actual. La medición manda. Este documento no
reabre la discusión; la registra para que dentro de seis meses nadie "corrija" el diseño
creyendo que fue un descuido.

Lo que el backend gana a cambio es todo supervision: annotators, ByteTrack (tracking con
`tracker_id`), `PolygonZone` y contadores de línea, `InferenceSlicer`, métricas — cosas
que el contrato actual `(N,6)` no tiene dónde poner.

### 1.2 Reparto por tipo de modelo

| tipo | quién dibuja | qué viaja por el WS |
|---|---|---|
| **detección** | backend (annotators) | frame compuesto, binario |
| **segmentación** | backend (annotators) | frame compuesto, binario |
| **clasificación** | **cliente — SIN CAMBIOS** | JSON `[{"cls","score"}]` |

Clasificación se queda exactamente como está (implementada el 2026-08-13): su resultado
es **texto, no geometría**, el panel HTML del cliente ya funciona, y componerlo en el
backend obligaría a re-encodear un frame entero para estampar tres renglones. Sería
gastar 100% del ancho de banda para transportar 40 bytes de información.

Segmentación es el caso donde la decisión es más obviamente correcta: mandar la máscara
cruda cuesta más que mandar el frame ya compuesto, y evita tener que elegir formato de
transporte de máscara (RLE vs PNG base64 vs bitmap reducido), que era la decisión de
diseño abierta del pendiente #7. **Esta decisión la mata**: la máscara nunca sale del
backend.

### 1.3 Dependencias

Este paso es el último de tres y **no se puede empezar antes**:

1. **Paso 1** — Python 3.12 + CUDA 12
   (`2026-08-13-migracion-python-312-cuda12-design.md`). supervision moderno pide ≥3.10;
   en 3.8 quedaríamos clavados a la 0.25.1 de 2024.
2. **Paso 2** — `sv.Detections` como tipo de dominio interno detrás del seam de `tasks/`.
   Los annotators **consumen `sv.Detections`**: sin el paso 2, este paso es dibujar a mano
   con `cv2.rectangle` y no gana nada.
3. **Paso 3** — este documento.

Antes de todo eso queda pendiente la verificación visual en Electron del panel de
clasificación, que es trabajo del árbol actual y no depende de nada de esto.

## 2. Alcance

### Entra

- Contrato nuevo del WebSocket: respuesta **binaria** para detección y segmentación,
  JSON para clasificación y para todos los errores (§3).
- `TaskStrategy` gana la noción de "esta tarea produce un frame" (§4.1).
- Composición + re-encode en el backend, con los annotators de supervision (§4.2).
- `drawSettings` viaja al backend: endpoint nuevo (§4.3).
- `label_map` en el schema de config: **deja de ser opcional** (§4.4).
- Bucket `draw_ms` en el PerfMeter (§4.5).
- Cliente: se borra el dibujo de cajas; `videoStream.ts` aprende a recibir binario (§5).

### NO entra (explícitamente)

- **Tracking, zonas, contadores.** Este paso deja la *cañería* del render en backend.
  Usar ByteTrack o `PolygonZone` es funcionalidad nueva y va después, ya barata.
- **Clasificación.** Ni una línea. Si al terminar este paso el panel de clasificación
  cambió de comportamiento, es un bug del paso.
- **Segmentación real.** El pipeline de SEG (unpacker + decode de máscara) sigue sin
  implementarse; este paso solo deja el camino de salida listo para cuando exista.
- **Empaquetado** (pendiente #20).

## 3. El contrato nuevo del WebSocket

Es el corazón del cambio y la única parte que rompe compatibilidad.

### 3.1 Hoy

`src/api/mainAPI.py:230` — el WS **siempre** responde un `send_json` con
`{task, result, error}`, un mensaje por frame recibido.

### 3.2 Después

El WS sigue respondiendo **siempre exactamente un mensaje por frame**, pero el mensaje
tiene dos formas posibles:

```
frame JPEG binario  ──WS──>  backend
                    <──WS──  BINARIO  = frame JPEG ya compuesto (detección, segmentación)
                    <──WS──  TEXTO    = envelope JSON {task, result, error}
                                        · clasificación (result = [{cls,score}])
                                        · CUALQUIER error, de cualquier tarea
                                          (task/result en null, como hoy)
```

El cliente discrimina por el tipo del dato recibido, no por un campo:

```ts
ws.onmessage = (event) => {
  if (typeof event.data === 'string') { /* envelope JSON: error o clasificación */ }
  else                                { /* Blob: frame compuesto, pintar y listo */ }
};
```

**Por qué binario y no base64 dentro del `result`.** Meter el JPEG como string base64
en el envelope es la opción tentadora porque no toca el contrato. Cuesta **+33% de
bytes por frame** y deshace la **Reforma 4**, que sacó el base64 del camino de entrada
justamente por eso. La medición que justifica este paso se hizo sobre el costo del
render, no sobre el de inflar cada frame un tercio. `websocket.send_bytes()` es una
línea y no tiene ese costo.

### 3.3 La invariante que no se puede romper

**Siempre una respuesta por frame recibido**, aunque el frame sea inválido, no haya
modelo o falle la inferencia. Romperla reintroduce el deadlock del stream (bug #3). El
cliente mantiene además su timeout de 3s (`videoStream.ts:90`) como red de seguridad.

Con el contrato nuevo esto sigue siendo cierto y es más fácil de auditar: **todo camino
de error responde JSON**; solo el camino feliz de det/seg responde binario.

### 3.4 Cómo sabe el cliente qué tarea está activa

Ya lo sabe: `workspaceStore.activeModel.type` se setea al seleccionar el modelo, antes
de que empiece el stream. El campo `task` del envelope deja de ser el mecanismo de
despacho del render y queda como dato de diagnóstico en los errores.

## 4. Cambios en el backend

### 4.1 `TaskStrategy`: de "serializar" a "presentar"

Hoy (`tasks/strategy.py`): `serialize: (result) -> dato JSON-listo`.

El problema es que componer un frame necesita **la imagen**, que `serialize` no recibe.
Dos campos nuevos en la dataclass frozen:

```python
@dataclass(frozen=True)
class TaskStrategy:
    task: str
    build_pipeline: Callable
    serialize: Callable          # (result) -> JSON. Se queda: la usa clasificación.
    output_kind: str = "json"    # "json" | "frame"
    render: Optional[Callable] = None   # (result, img_bgr, draw_cfg) -> bytes (JPEG)
```

- `detection` / `segmentation`: `output_kind="frame"`, `render` implementado.
- `classification`: `output_kind="json"`, `render=None`. **Sin tocar.**

El WS despacha por `output_kind`, no por `task` — así agregar un tipo nuevo sigue siendo
"registrar una estrategia" y el handler del WS no vuelve a crecer un `if` por tipo.

`render` recibe el `img_bgr` **ya decodificado** que el handler tiene en la mano
(`mainAPI.py:252`), así que no hay decode extra.

### 4.2 La composición

En `tasks/detection.py`, alimentado por el `sv.Detections` que produce el paso 2:

```python
annotated = box_annotator.annotate(scene=img_bgr.copy(), detections=dets)
annotated = label_annotator.annotate(scene=annotated, detections=dets, labels=labels)
ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
```

Los annotators se construyen **una vez al armar el pipeline** (son objetos con estado de
configuración, no funciones puras), coherente con el estilo de builders/closures del
repo. `scene=img_bgr.copy()` no es opcional: los annotators de supervision escriben
in-place.

**Doble compresión.** El frame llega ya comprimido a `q=0.8` (`videoStream.ts:127`) y
sale comprimido otra vez. Es pérdida de calidad, no de latencia — un eje distinto del
que se midió. `quality` va como parámetro configurable (default 85) para poder subirlo
si se ve degradación en el feed.

### 4.3 `drawSettings` tiene que viajar al backend

Hoy vive en el cliente (`workspaceStore.ts`, persistido en `localStorage`:
`bboxColor`, `labelColor`, `maskAlpha`). Si el backend dibuja, el backend necesita esos
valores. Endpoint nuevo — es, en los hechos, la resurrección del `/config/colors` que se
había eliminado:

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/config/draw` | `{bboxColor, labelColor, maskAlpha, thickness?}` → ajustes de dibujo en vivo. 422 fuera de rango o color inválido |

- El cliente sigue siendo dueño del estado y de su persistencia; hace push al backend
  **al cambiar un ajuste y al cargar un modelo** (por si el backend se reinició).
- Se aplica **al frame siguiente**, no al actual. Se pierde el cambio de color
  instantáneo que daba el dibujo client-side: costo conocido y aceptado.
- Los colores llegan como `#RRGGBB` y se convierten a BGR en el backend. Validar en el
  endpoint, no en el hot path.

### 4.4 `label_map` deja de ser opcional

Nota abierta de §6 de CLAUDE.md: `DetectionOutput` no tiene `label_map` y las etiquetas
muestran el id numérico. Mientras dibujaba el cliente era un defecto cosmético
postergable. **Ahora es requisito del paso**: los annotators dibujan lo que se les da, y
sin nombres cada frame va a decir `0`, `17`, `663`.

Campo nuevo en el schema (`output`), opcional en el JSON pero consumido de verdad:

```json
"label_map": ["person", "bicycle", "car", "..."]
```

Lista indexada por `class_id` (la forma natural: es lo que traen los `.names` de casi
todos los modelos). Ausente → se dibuja el id numérico, como hoy. El schema es estricto
(`extra="forbid"`), así que agregar el campo obliga a tocar también el wizard — el
recordatorio de siempre (pendiente #9).

Beneficio lateral: el `label_map` **no viaja al cliente**, se resuelve donde se dibuja.
Es más barato que la alternativa que había que diseñar antes.

### 4.5 Métricas

El PerfMeter mide `pre_ms / inf_ms / post_ms`. El anotado + re-encode caería dentro de
`post_ms` y quedaría invisible: exactamente el número que hace falta para defender la
medición que motivó este paso. Bucket nuevo **`draw_ms`** en los `timings` que devuelve
el runner, propagado al `/metrics`.

## 5. Cambios en el cliente

Es, sobre todo, **borrar**.

- **`videoStream.ts:72`** — `ws.onmessage` hoy hace `JSON.parse(event.data as string)` y
  descarta lo que no parsea. Aprende la rama binaria: `Blob` → `createImageBitmap` →
  entregar al consumidor para pintar. `waitingForResponse = false` se sigue soltando en
  ambas ramas (anti-deadlock).
- **`present.ts`** — se simplifica mucho: para det/seg ya no hay parse ni estrategia, se
  pinta el bitmap recibido. La lógica de `releasePrevious()` / `strategy.clear()`
  (agregada el 2026-08-13) **se conserva**: sigue haciendo falta para desmontar el panel
  HTML de clasificación al cambiar de modelo.
- **`detection.service.ts`** — el dibujo de cajas se elimina. La estrategia queda como
  entrada del registry con `output_kind` "frame" o directamente se retira; decisión de
  implementación, no de diseño.
- **`segmentation.service.ts`** — sigue siendo stub, pero ahora ya no le espera un
  pipeline de decode de máscara en el cliente. Nunca lo va a tener.
- **`classification.service.ts`** — **no se toca.**
- **`DrawSettingsModal.tsx`** — mismo UI; el `setDrawSettings` ahora además hace push a
  `POST /config/draw`.

### 5.1 El espejo NO es un problema (verificado)

Riesgo que se planteó y **se descartó leyendo el código**: si el cliente espejara el
frame *compuesto*, los textos de las etiquetas saldrían al revés. No pasa:
`videoStream.ts:105` aplica el espejo **en la captura**, sobre el canvas, *antes* de
encodear el JPEG que se envía. El backend recibe el frame ya espejado, compone encima, y
vuelve correcto. No hay nada que hacer acá — solo **no** agregar un espejo de display
"por las dudas".

## 6. Lo que NO cambia (y conviene decirlo)

- **Los unpackers, `anchor_gen.py`, el `output_adapter` y el undo del letterbox se
  quedan.** supervision es un **contenedor**, no un intérprete: `sv.Detections` exige
  `xyxy`/`confidence`/`class_id` ya calculados, y sus `from_*` cubren librerías concretas
  (ultralytics, transformers, detectron2…), nunca un tensor crudo de ONNX/TFLite descrito
  por un JSON. Eso es precisamente lo que hace UNCaLens y por eso no es reemplazable.
- **El NMS propio se queda.** `sv.Detections.with_nms()` va a tentar. No: el nuestro está
  probado y atado a rarezas por formato (`tflite_detpost` ya trae NMS y umbral aplicados
  por el op de TFLite, y el postproceso los desactiva para ese caso). Cambiarlo es riesgo
  sin ganancia.
- **El `ModelController` sigue siendo manager puro.** Todo lo nuevo vive detrás del seam
  de `tasks/`.
- **Un frame en vuelo**, el timeout de 3s y "siempre una respuesta" siguen igual.

## 7. Riesgos

1. **El threadpool ahora hace más trabajo por frame.** El `run_in_executor`
   (`mainAPI.py:262`) suma anotado + encode. Con inferencia en GPU a ~14 ms, un encode de
   1080p (~5-10 ms) deja de ser despreciable en proporción aunque sea invisible en
   latencia percibida. Es exactamente lo que el bucket `draw_ms` va a mostrar.
2. **Pérdida de calidad acumulada** por la doble compresión (§4.2). Mitigable con
   `quality`; verificar a ojo en el feed.
3. **El cliente pierde los datos numéricos.** Hoy recibe las cajas y podría, por ejemplo,
   listarlas o contarlas. Después del cambio solo recibe píxeles. Si alguna vez hace falta
   el dato *además* del frame, la salida es mandar el envelope JSON **y** el binario
   (dos mensajes por frame) — lo que rompe la invariante de §3.3 tal como está escrita.
   **No hacerlo en este paso**; anotado porque es la evolución previsible.
4. **Depende del paso 2.** Si el paso 2 se salta, esto degenera en dibujar con
   `cv2.rectangle` a mano y no queda nada de la ganancia.

## 8. Verificación (en orden)

1. **Tests**: `pytest` verde. Los del envelope de detección **van a tener que cambiar**
   (ya no hay `result` con cajas); los de clasificación **no deben tocarse** — si un test
   de CLS necesita cambiar, algo se rompió.
2. **Clasificación intacta**: cargar `saved_model_class`, verificar envelope
   `{"task":"classification","result":[{"cls":663,...}]}` idéntico al de hoy y panel
   funcionando.
3. **Detección**: cargar `yolov7-tiny`, mandar `horses.jpg`, recibir binario, y que el
   frame que vuelve tenga las 5 cajas dibujadas **con nombres de clase**, no ids.
4. **Errores**: sin modelo → JSON `no_model`; frame corrupto → JSON `frame_invalido`;
   inferencia rota → JSON `inference_error`. Los tres, con el stream corriendo, sin que
   el cliente se cuelgue.
5. **Ida y vuelta**: detección → clasificación → detección sin residuos en pantalla (es
   el bug que se arregló el 2026-08-13 con `strategy.clear()`).
6. **Métricas**: `/metrics` muestra `draw_ms` con valores razonables. **Anotar el número
   y compararlo con la medición previa** — es la evidencia de la decisión, conviene
   dejarla en el repo y no en la memoria de nadie.
7. **Cliente**: `npm run typecheck` y `npm run build`.
8. **Electron**: `npm start`, inferencia sobre archivo, feed con cajas dibujadas por el
   backend.

## 9. Rollback

Es una rama, y el dibujo del cliente queda en el historial de git. El punto de no retorno
suave es el borrado de `detection.service.ts`: hasta ahí, volver es un `git revert`.

## 10. Criterio de terminado

- El cliente no contiene ni una línea que dibuje una caja.
- El WS responde binario para det/seg y JSON para cls/errores, siempre un mensaje por
  frame.
- Clasificación se comporta exactamente igual que antes del paso.
- `label_map` se consume y las etiquetas muestran nombres.
- `/metrics` expone `draw_ms`.
- CLAUDE.md actualizado: §4 (el diagrama del flujo cambia), §5 (endpoint nuevo), y una
  nota en §7 de que la Reforma 3 fue revertida **a propósito y con medición**.
