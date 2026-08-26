# Paso 2: `sv.Detections` como tipo de dominio interno

- **Fecha**: 2026-08-26
- **Rama**: `main`
- **Estado**: **HECHO**
- **Contexto**: paso 2 de los tres del pendiente #23 (decidido el 2026-08-21). El paso 1
  (Python 3.12 + CUDA 12) se cerró el 2026-08-25. El paso 3 es
  `2026-08-21-render-backend-supervision-design.md`, que **depende de este**.

## 1. Qué se hizo, y por qué es un paso propio

Los annotators, ByteTrack y las zonas de supervision **consumen `sv.Detections`**. Si el
paso 3 empezara con un `(N,6)` pelado en la mano, sería dibujar con `cv2.rectangle` a
mano y no quedaría nada de la ganancia (riesgo 4 del spec del paso 3). Así que primero se
cambia el **tipo de dominio** y después se dibuja.

Se hizo **sin tocar el contrato del WebSocket**: el cliente recibe exactamente las mismas
filas `[x1,y1,x2,y2,conf,cls]` que antes. Este paso es puramente interno y por eso es
verificable contra la salida anterior, byte a byte. El que rompe el contrato es el paso 3.

## 2. El cambio

```
antes:  unpack -> adapter -> postprocess -> List[List[float]] -> serialize -> envelope
ahora:  unpack -> adapter -> postprocess -> ndarray (N,6) -> sv.Detections -> serialize -> envelope
                                            └── tasks/domain.py ──┘
```

- **`src/api/func/tasks/domain.py`** (nuevo): `detections_from_array()`,
  `array_from_detections()`, `empty_detections()`. Es **la única frontera** entre el
  `(N,6)` y el tipo de dominio; el resto del pipeline (unpackers, `output_adapter`,
  `anchor_gen`, NMS, undo del letterbox) sigue hablando ndarray y **no se enteró**.
- **`tasks/detection.py`**: el runner devuelve `sv.Detections`; `serialize_detection`
  lo consume vía `array_from_detections` y produce el mismo JSON de siempre. Es estricto:
  si le pasan un ndarray levanta `TypeError` (el `(N,6)` dejó de ser el tipo de dominio y
  no debe colarse por la puerta de atrás).
- **`output_transformer.buildPostprocessor`**: dejó de hacer `.tolist()`. Devuelve
  `ndarray (N,6) float32`. **Hallazgo**: CLAUDE.md decía que el postproceso devolvía un
  ndarray `(N,6)`, pero devolvía una **lista de listas** — una copia a objetos Python por
  frame, en pleno hot path, contradiciendo la regla de "nunca `.tolist()`" que el mismo
  documento le exige a los `predict_fn`.
- **`empty_detections()` en vez de `sv.Detections.empty()`**: el `.empty()` de supervision
  deja `confidence`/`class_id` en `None` y obliga a chequear `None` en cada consumidor.
  Con arrays de largo 0 el código de abajo es el mismo con 0 o con N cajas.

## 3. Lo que NO cambió (a propósito)

- **El envelope del WS y el cliente**: ni una línea. Eso es del paso 3.
- **Clasificación**: su resultado es `(K,2)` y no es geometría. `sv.Detections` no le
  aporta nada. Sin tocar.
- **Segmentación**: sigue en 501. Lo que gana es lugar donde poner la máscara
  (`sv.Detections.mask`) cuando exista el decode.
- **El NMS propio, los unpackers, `anchor_gen`, el undo del letterbox**: se quedan, por lo
  ya argumentado en §6 del spec del paso 3 — supervision es un contenedor, no un
  intérprete de tensores crudos.

## 4. Costo medido

Conversión ida y vuelta (`detections_from_array` + `array_from_detections` + serialize)
contra el camino viejo (`.tolist()` + serialize), 2000 corridas:

| cajas | viejo | nuevo | delta |
|---|---|---|---|
| 6 (caso real) | 0,0115 ms | 0,0221 ms | **+0,011 ms/frame** |
| 100 | 0,187 ms | 0,255 ms | +0,069 ms/frame |

Contra ~7,4 ms de inferencia en GPU es ruido. Medido end-to-end sobre el backend real:
`post_avg_ms = 0,159` sobre 31 frames de `yolov7-tiny` (avg total 11,8 ms, 84 fps).

## 5. Dependencia nueva

`supervision==0.30.1`. Arrastra `scipy`, `av`, `tqdm`, `defusedxml`, `pyDeprecate`
(~40 MB). **No pisó nada** del entorno recién migrado: el diff de
`requirements.lock.txt` es puramente aditivo (numpy 2.5.2, opencv 5.0, ORT 1.26 y TF 2.21
quedaron donde estaban).

## 6. Verificación

- `pytest`: **83 verdes** (82 + un test nuevo de que `serialize_detection` rechaza lo que
  no es `sv.Detections`).
- Backend real por HTTP+WS, los tres modelos, contra la línea de base del 2026-08-25:
  `yolov7-tiny` 6 cajas clase 17, `saved_model_class` 663/813, `efficientdet-lite0` 3
  cajas. Envelope **idéntico** al de antes del cambio.
- Invariante "el WS siempre responde": 30 frames seguidos OK, frame corrupto →
  `frame_invalido` y el stream **sigue vivo**, sin modelo → `no_model`.
- El cliente no se tocó y no hacía falta tocarlo: el envelope no cambió.

## 7. Qué destraba

El paso 3 ya puede escribirse como está especificado: `box_annotator.annotate(scene=...,
detections=dets)` recibe directamente lo que el runner devuelve. Y ByteTrack/`PolygonZone`
(que el spec del paso 3 deja explícitamente afuera) pasan a ser baratos de agregar después.
