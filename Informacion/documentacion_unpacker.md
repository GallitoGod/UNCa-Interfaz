# Documentación técnica – Catálogo de contratos de salida y desempaquetadores

**Proyecto:** UNCa‑Interfaz
**Autor:** Pablo Gallo
**Fecha de creación:** 2026‑01‑26
**Versión del documento:** v1.1
**Última actualización:** 2026‑06‑19

> **Nota de cambio v1.0 → v1.1 (2026‑06‑19):** se alineó la documentación con el
> código real tras las reformas 6 y 8. Cambios principales:
> - El IR implementado es una **matriz `(N, 6) float32`**, no una estructura con
>   `boxes/scores/classes` separados.
> - Se agregaron al catálogo los contratos **Anchor Deltas** y **Raw**,
>   antes solo descritos parcialmente.
> - Los flujos de desempaquetado ahora reflejan la salida real de **6 columnas con
>   `class_id`** (antes documentaban 5 sin clase).
> - Se documentó la firma real `fn(raw_output, sh=None)` y la eliminación del estado
>   por frame de `runtimeShapes` (reforma 8).
> - Se corrigió la numeración duplicada (dos "6.1") y se formalizó el registry real.

---

## 1. Propósito del documento

Este documento tiene como objetivo registrar, explicar y formalizar la arquitectura conceptual y técnica relacionada con:

* Los contratos de salida de modelos de Aprendizaje Automático (detección).
* El sistema de desempaquetadores (unpackers) asociados a cada contrato.
* El enfoque de clusterización de salidas, evitando la falsa universalidad.

El documento existe para:


* Dejar constancia de decisiones de diseño.
* Evitar la re‑introducción de ideas descartadas (ej. desempaquetador genérico universal).
* Servir como base de evolución futura del sistema.

---

## 2. Principio fundamental de diseño

> **Un modelo no devuelve datos: devuelve resultados de un grafo de computo.**

Las salidas de un modelo de AA son la manifestación directa de su grafo de cómputo, entonces:

* No existe una salida universal.
* No existe una semántica implícita común.
* Toda salida debe interpretarse en función del modelo que la genera.

Esto invalida la idea de un desempaquetador genérico único.

---

## 3. Enfoque adoptado: clusterización por contratos de salida

El sistema adopta un enfoque de catálogo de contratos de salida, donde:

* Cada contrato representa una familia semántica de outputs.
* Cada contrato posee:

  * Forma esperada de tensores
  * Significado de ejes
  * Supuestos implícitos (anchors, grillas, NMS, etc.)
  * Un desempaquetador específico

Todos los contratos convergen en un IR interno común.

---

## 4. Artefacto central: el IR de detección (matriz `(N, 6)`)

### 4.1 Identificador

**Artefacto:** `DetectionIR`
**ID:** IR‑DET‑001
**Versión:** v1.1

### 4.2 Rol

Estructura intermedia que representa detecciones independientemente del modelo de
origen. Es lo único que el resto del pipeline (output_adapter → output_transformer)
consume después de cada desempaquetador.

### 4.3 Estructura **implementada**

Conceptualmente el IR son detecciones `(boxes, scores, classes)`, pero en el código
**no es una estructura de objetos**: por rendimiento (hot path, sin objetos Python)
cada unpacker devuelve una **matriz NumPy `(N, 6) float32`**:

```
[ c0, c1, c2, c3, conf, class_id ]
```

* `c0..c3` — las cuatro coordenadas de la caja, **en el layout propio del unpacker**
  (no necesariamente `xyxy` todavía):
  * `yolo_flat`  → `[cx, cy, w, h]`
  * `boxes_scores`, `tflite_detpost`, `anchor_deltas` → ya emiten `[x1, y1, x2, y2]`
    salvo donde se indique (ver §6 por cada flujo).
* `conf` — score final ya combinado (objectness × prob, o prob de la mejor clase).
* `class_id` — id de clase como `float32` (se castea a entero aguas abajo).

### 4.4 Del IR al formato estándar del sistema

El **output_adapter** reordena las columnas `c0..c3` al contrato estándar del sistema
`[x1, y1, x2, y2, conf, cls]`. Esto **solo** se aplica a los formatos del conjunto
`_NEEDS_ADAPTER = {"raw", "yolo_flat", "tflite_detpost", "anchor_deltas"}`
(definido en `model_controller.py`). `boxes_scores` **ya** entrega `[x1, y1, x2, y2]`,
por lo que aplicarle el adapter re‑intercambiaría x/y.

Luego el **output_transformer** aplica filtro de confianza → top‑k → NMS → undo
letterbox → orden por score, dejando las cajas en **píxeles de la imagen original**.

> **Deuda conocida ("lazy scaling"):** el contrato de normalización está repartido
> entre unpacker y postproceso. Cada unpacker decide si escala a píxeles del tensor
> según `runtime.runtimeShapes.out_coords_space`. Tocar con cuidado (ver §6 y los
> comentarios en `utils.py`).

---

## 5. Catálogo de contratos de salida

> Todos los contratos se despachan vía `UNPACKERS_REGISTRY` (registry.py) y devuelven
> la matriz `(N, 6) float32` descrita en §4.3.

### 5.1 Contrato YOLO‑Flat

**ID:** CONTRACT‑DET‑YOLO‑001
**Versión:** v1.0
**Fecha:** 2026‑01‑26
**Builder:** `build_yolo_flat` (`yolo_flat.py`)

#### Descripción

Salida típica de modelos YOLO (v3–v8 y derivados), con regresión directa sobre grilla.

#### Forma típica

* Tensor único `(B, N, 5 + C)` o variantes

#### Semántica

* `(cx, cy, w, h)`
* `objectness`
* `class_scores`

---

### 5.2 Contrato Boxes + Scores

**ID:** CONTRACT‑DET‑BOXES‑002
**Versión:** v1.0
**Fecha:** 2026‑01‑26
**Builder:** `build_boxes_scores` (`boxes_scores.py`)

#### Descripción

Modelo devuelve tensores separados para cajas y puntuaciones.

#### Forma típica

* `boxes`: `(N, 4)` en orden `[ymin, xmin, ymax, xmax]`
* `scores`: `(N, C)`

#### Supuestos

* NMS externo al modelo.
* El unpacker detecta cuál tensor es el de cajas por su última dimensión (`== 4`).
* **Único contrato que NO pasa por el output_adapter**: ya entrega `[x1, y1, x2, y2]`.

---

### 5.3 Contrato TFLite DetectionPostProcess

**ID:** CONTRACT‑DET‑TFLITE‑003
**Versión:** v1.0
**Fecha:** 2026‑01‑26
**Builder:** `build_tflite_detpost` (`tflite_detpost.py`)

#### Descripción

Modelo incluye postprocesamiento (NMS) dentro del grafo.

#### Outputs

* `detection_boxes`   `(1,N,4)` o `(N,4)` — `[ymin, xmin, ymax, xmax]`
* `detection_scores`  `(1,N)` o `(N,)`
* `detection_classes` `(1,N)` o `(N,)`
* `num_detections`    `(1,)` — opcional (recorta a `N = min(count, boxes.shape[0])`)

#### Características

* Boxes normalizadas (por defecto `out_coords_space = "tensor_pixels"`).
* NMS y umbral **embebidos**: el postproceso los desactiva por defecto para este formato.

---

### 5.4 Contrato Anchor Deltas (EfficientDet / SSD crudos)

**ID:** CONTRACT‑DET‑ANCHOR‑004
**Versión:** v1.0
**Fecha:** 2026‑01‑31
**Builder:** `build_anchor_deltas` (`anchor_deltas.py`)

#### Descripción

Cabeza cruda anchor‑based: el modelo exporta deltas de caja + logits/probas de clase,
**sin** `DetectionPostProcess`. Es el contrato más acoplado al modelo.

#### Forma típica

* `box_deltas`:   `(1,N,4)` o `(N,4)` — `[ty, tx, th, tw]`
* `class_scores`: `(1,N,C)` o `(N,C)` — logits o probabilidades

#### Supuestos / dependencias en runtime

* `runtime.runtimeShapes.anchors` `(N,4)` normalizados `[ay, ax, ah, aw]`.
* `runtime.runtimeShapes.box_variance` `(4,)`, típicamente `[0.1, 0.1, 0.2, 0.2]`.
* La tabla de anchors **NO viaja en el JSON**: el controller la genera al cargar con
  `anchor_gen.generate_efficientdet_anchors()` a partir de `output.anchor_config`.
* `scores_activation` (en `anchor_config`) declara cómo activar el tensor de clases:
  * `"none"`    — el tensor ya trae probabilidades (caso EfficientDet‑lite TFLite).
  * `"sigmoid"` — logits sigmoideos.
  * `"softmax"` — logits softmax (default si no hay `anchor_config`).
* Siempre emite **píxeles del tensor**: el JSON debe declarar
  `out_coords_space = "tensor_pixels"`.

> El orden de aplanado de los anchors debe coincidir con el head del modelo:
> niveles asc > posiciones fila‑major > (octava externa, aspect interno).

---

### 5.5 Contrato Raw (fallback / depuración)

**ID:** CONTRACT‑DET‑RAW‑005
**Versión:** v1.0
**Fecha:** 2026‑01‑26
**Builder:** `build_raw` (`raw.py`)

#### Descripción

Desempaquetador de respaldo y depuración. **Es el fallback** cuando `pack_format` no
está registrado o no se declara. No interpreta semántica: solo normaliza la forma del
tensor a 2D.

#### Comportamiento

* Acepta un único tensor (o lista/tupla de longitud 1).
* `(1, N, F) → (N, F)`; `(F,) → (1, F)`; lista vacía → `(0, 6)`.
* Lanza `ValueError` si recibe más de un output (no sabe cuál usar).
* Pasa por el output_adapter (`raw ∈ _NEEDS_ADAPTER`).

---

## 6. Sistema de desempaquetadores

### 6.0 Firma y diagramas de flujo

Todos los unpackers se construyen con un patrón **builder/closure**:
`build_<formato>(output_cfg)` devuelve `fn(raw_output, sh=None) -> np.ndarray (N,6)`.

* `raw_output` — los tensores crudos del backend (numpy, nunca `.tolist()`).
* `sh` — el objeto `runtime`; el helper `rt_shapes(sh)` extrae `runtime.runtimeShapes`.

**Reforma 8 — sin estado mutable por frame:** `runtimeShapes` solo contiene
**constantes de carga** (`input_width/height`, `out_coords_space`, `anchors`,
`box_variance`). El estado por frame (`orig_width/height`, scale/pads del letterbox)
viaja en el dict `meta` que produce `preprocess_fn` y consume `postprocess_fn` — el
unpacker no lo toca. Cada inferencia es autocontenida.

Cada contrato documentado a continuación posee un diagrama de flujo explícito que
describe el recorrido semántico desde la salida cruda del modelo hasta el IR `(N,6)`.
Estos diagramas no representan código, sino decisiones de interpretación del grafo de
cómputo. Su función es servir como mapa mental rápido, evitar ambigüedades al retomar
el proyecto, y justificar por qué cada contrato necesita su propio desempaquetador.

---

### 6.1 Contrato YOLO-Flat – Flujo de desempaquetado

**ID:** FLOW-DET-YOLO-001
**Versión:** v1.1
**Fecha:** 2026-06-19

**Resumen del flujo:**

1. Recepción de tensor crudo `(N, 5 + C)` con formato:
   `[cx, cy, w, h, obj, p0, p1, …, pC]`.
2. Normalización estructural (`to_2d`) para garantizar matriz 2D. Si tiene menos de 6
   columnas o está vacío → `(0, 6)`.
3. Separación semántica de columnas: `(cx, cy, w, h)`, `objectness`, probabilidades.
4. Selección de clase dominante: `best_cls = argmax(cls_probs)`.
5. Cálculo de score final: `score = objectness * max(cls_probs)`.
6. Decisión de espacio de coordenadas:
   * si `runtime.out_coords_space == "tensor_pixels"` → escala `cxcywh` a píxeles del
     tensor (`scale_cxcywh_inplace`); en caso contrario se conserva tal cual.
7. Salida `(N,6)`: `[cx, cy, w, h, score, class_id]` (el output_adapter la reordena
   a `xyxy` después).

Este flujo asume regresión directa, sin anchors ni NMS embebido.

---

### 6.2 Contrato Boxes + Scores – Flujo de desempaquetado

**ID:** FLOW-DET-BOXES-002
**Versión:** v1.1
**Fecha:** 2026-06-19

**Resumen del flujo:**

1. Recepción de dos tensores crudos: uno `(N, 4)` (cajas), otro `(N, C)` (scores).
2. Detección dinámica de cuál tensor representa cajas (última dimensión `== 4`).
3. Normalización estructural (`to_2d`) de ambos tensores.
4. Separación semántica — cajas en orden `(ymin, xmin, ymax, xmax)`, scores por clase.
5. Selección de clase dominante: `best_cls = argmax(scores, axis=1)`, `best_p`.
6. Decisión de espacio de coordenadas:
   * si `out_coords_space == "tensor_pixels"` → escala `xyxy` a píxeles del tensor.
7. Salida `(N,6)`: **`[x1, y1, x2, y2, best_p, class_id]`** — el unpacker **ya
   reordena** de `yxyx` a `xyxy`, por eso **no** pasa por el output_adapter.

Este contrato no incluye NMS, que se aplica posteriormente en el postproceso.

---

### 6.3 Contrato TFLite DetectionPostProcess – Flujo de desempaquetado

**ID:** FLOW-DET-TFLITE-003
**Versión:** v1.1
**Fecha:** 2026-06-19

**Resumen del flujo:**

1. Recepción de outputs múltiples: `boxes`, `scores`, `classes` y `count` (opcional).
2. Normalización estructural de `boxes` (`to_2d`).
3. Determinación del número efectivo de detecciones:
   * si `count` existe → `N = min(count, boxes.shape[0])`; si no → `N = boxes.shape[0]`.
4. Recorte coherente de `boxes`, `scores` y `classes` a `N`. Si `N <= 0` → `(0,6)`.
5. Separación semántica de cajas: `(ymin, xmin, ymax, xmax)`.
6. Decisión de espacio de coordenadas:
   * **default `out_coords_space = "tensor_pixels"`** → escala `xyxy` a píxeles.
7. Salida `(N,6)`: `[ymin, xmin, ymax, xmax, score, class_id]` (el output_adapter la
   reordena a `xyxy`).

Este contrato ya incluye NMS y filtrado, embebidos en el grafo; el postproceso los
desactiva por defecto.

---

### 6.4 Contrato Anchor Deltas – Flujo de desempaquetado

**ID:** FLOW-DET-ANCHOR-004
**Versión:** v1.1
**Fecha:** 2026-06-19

**Resumen del flujo:**

1. Recepción de dos tensores crudos: `box_deltas` y `class_scores`.
2. Validación de dependencias externas: `runtime.anchors` y `runtime.box_variance`
   (si falta variance, default `[0.1, 0.1, 0.2, 0.2]`).
3. Detección de cuál tensor contiene deltas (última dimensión `== 4`); validación de
   que `N anchors == N deltas`.
4. Activación de `class_scores` según `scores_activation`: `softmax` / `sigmoid` /
   `none` (ya probas).
5. Selección de clase dominante: `best_cls = argmax(class_probs)`, `best_p`.
6. Decodificación geométrica `anchor + deltas → yxyx` normalizado
   (`decode_anchor_deltas_to_yxyx`).
7. Escalado a píxeles del tensor `(W, H)` (siempre — emite `tensor_pixels`).
8. Salida `(N,6)`: `[ymin, xmin, ymax, xmax, best_p, class_id]` (el output_adapter la
   reordena a `xyxy`).

Este contrato es el más acoplado al modelo, ya que depende de la tabla de anchors
generada en carga.

---

### 6.5 Principio

> **Cada desempaquetador conoce un contrato y solo uno.**

No existe lógica condicional genérica dentro de los unpackers.

### 6.6 Dispatch

La selección del desempaquetador se realiza mediante:

* Configuración del modelo (`output.pack_format` en el JSON).
* Tipo explícito definido en JSON.

Nunca por heurísticas implícitas sobre el contenido del tensor.

---

## 7. Extensibilidad: Registry de Unpackers

Esta sección formaliza el patrón para escalar el catálogo de contratos sin crecer en
complejidad accidental.

### 7.1 Patrón: `UNPACKERS_REGISTRY`

**ID:** EXT-PATTERN-REG-001
**Versión:** v1.1
**Fecha:** 2026-06-19

* **Problema:** un `if/elif` grande en el selector de desempaquetadores tiende a crecer
  sin control, mezclar lógica de selección con lógica de desempaquetado, y volver
  costosa la extensión (se "toca el núcleo" para agregar formatos).

* **Solución:** un registry explícito (diccionario) que mapea `pack_format → builder`.

* **Beneficios:** agregar un contrato nuevo = escribir el builder + registrarlo (sin
  tocar el controlador); facilita pruebas unitarias por contrato; reduce ruido en
  `unpack_out`.

**Implementación real (`registry.py`):**

```python
UNPACKERS_REGISTRY = {
    "raw":            build_raw,
    "yolo_flat":      build_yolo_flat,
    "boxes_scores":   build_boxes_scores,
    "tflite_detpost": build_tflite_detpost,
    "anchor_deltas":  build_anchor_deltas,
}

def unpack_out(output_cfg):
    fmt = (getattr(output_cfg, "pack_format", None) or "raw").lower()
    factory = UNPACKERS_REGISTRY.get(fmt, build_raw)   # fallback seguro a "raw"
    return factory(output_cfg)
```

* **Nota:** el fallback `"raw"` debe seguir existiendo para depuración/control.

### 7.2 Checklist para agregar un contrato nuevo

Agregar un `pack_format` toca **3 lugares** (ver reforma pendiente #9 en CLAUDE.md):

1. El builder en `unpackers/<formato>.py` + alta en `UNPACKERS_REGISTRY`.
2. El `Literal` de `pack_format` en `config_schema.py`.
3. El `<select>` del wizard en `configBuilder.js`.

Además hay que decidir si el formato entra en `_NEEDS_ADAPTER` (`model_controller.py`):
si el unpacker ya entrega `[x1, y1, x2, y2, …]`, **no** debe pasar por el output_adapter.

---

## 8. Versionado y trazabilidad

Cada artefacto del sistema debe poseer:

* ID único
* Versión semántica (`vMAJOR.MINOR`)
* Fecha de modificación
* Nota de cambio

Ejemplo:

```
CONTRACT‑DET‑YOLO‑001
v1.1 – 2026‑03‑02
Cambio: soporte para salida multi‑head
```

---

## 9. Decisiones descartadas (registro histórico)

### 9.1 Desempaquetador genérico universal

**Estado:** descartado

**Motivo:**

* Violación de semántica
* Penalización severa de rendimiento
* Imposibilidad de inferir significado sin contexto del grafo

---

## 10. Estado actual del sistema

* Arquitectura basada en contratos.
* Clusterización explícita vía `UNPACKERS_REGISTRY`.
* IR común implementado como matriz `(N, 6) float32`.
* Catálogo formalizado: `raw`, `yolo_flat`, `boxes_scores`, `tflite_detpost`,
  `anchor_deltas`.
* Soporte real de modelos anchor‑based crudos (EfficientDet/SSD) con tabla de anchors
  generada en carga.
* Sin estado mutable por frame en los unpackers (reforma 8).
* Documentación por contrato alineada con el código.

### Deuda abierta

* **Lazy scaling:** el contrato de normalización está repartido entre unpacker y
  postproceso (`out_coords_space` + escalado en `utils.py`). Unificar en una sola capa.
* La normalización de shapes en `inference()` duplica lo que hace `raw.py`; debería
  vivir en una sola capa (contrato del unpacker: "siempre `(N,6)` float32").

---

## 11. Notas finales

El objetivo es preservar el criterio de diseño frente al paso del tiempo y futuras
modificaciones.

---

**Fin del documento – v1.1**