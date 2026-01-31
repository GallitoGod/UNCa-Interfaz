# Documentación técnica – Catálogo de contratos de salida y desempaquetadores

**Proyecto:** UNCa‑Interfaz
**Autor:** Pablo Gallo
**Fecha de creación:** 2026‑01‑26
**Versión del documento:** v1.0

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

## 4. Artefacto central: DetectionIR

### 4.1 Identificador

**Artefacto:** `DetectionIR`
**ID:** IR‑DET‑001
**Versión:** v1.0

### 4.2 Rol

Estructura intermedia semántica que representa detecciones independientemente del modelo de origen.

### 4.3 Estructura conceptual

* `boxes`: ndarray (N, 4)
* `scores`: ndarray (N)
* `classes`: ndarray (N)
* `coords_type`: enum (`xyxy`, `cxcywh`, etc.)
* `normalized`: bool
* `metadata`: dict (opcional)

Este IR es el único formato consumido por el resto del sistema.

---

## 5. Catálogo de contratos de salida

### 5.1 Contrato YOLO‑Flat

**ID:** CONTRACT‑DET‑YOLO‑001
**Versión:** v1.0
**Fecha:** 2026‑01‑26

#### Descripción

Salida típica de modelos YOLO (v3–v8 y derivados), con regresión directa sobre grilla.

#### Forma típica

* Tensor único `(B, N, 5 + C)` o variantes

#### Semántica

* `(x, y, w, h)`
* `objectness`
* `class_scores`

#### Unpacker asociado

* `yolo_flat_unpacker`

---

### 5.2 Contrato Boxes + Scores

**ID:** CONTRACT‑DET‑BOXES‑002
**Versión:** v1.0
**Fecha:** 2026‑01‑26

#### Descripción

Modelo devuelve tensores separados para cajas y puntuaciones.

#### Forma típica

* `boxes`: `(N, 4)`
* `scores`: `(N, C)`

#### Supuestos

* NMS externo al modelo

#### Unpacker asociado

* `boxes_scores_unpacker`

---

### 5.3 Contrato TFLite DetectionPostProcess

**ID:** CONTRACT‑DET‑TFLITE‑003
**Versión:** v1.0
**Fecha:** 2026‑01‑26

#### Descripción

Modelo incluye postprocesamiento (NMS) dentro del grafo.

#### Outputs

* `detection_boxes`
* `detection_classes`
* `detection_scores`
* `num_detections`

#### Características

* Boxes normalizadas
* NMS embebido

#### Unpacker asociado

* `tflite_detpost_unpacker`

---

## 6. Sistema de desempaquetadores

### 6.0 Diagramas de flujo por contrato

Cada contrato documentado a continuación posee un diagrama de flujo explícito que describe el recorrido semántico desde la salida cruda del modelo hasta el IR interno.

Estos diagramas no representan código, sino decisiones de interpretación del grafo de cómputo.

Su función es:

* servir como mapa mental rápido,
* evitar ambigüedades al retomar el proyecto,
* y justificar por qué cada contrato necesita su propio desempaquetador.

---

### 6.1 Contrato YOLO-Flat – Flujo de desempaquetado

**ID:** FLOW-DET-YOLO-001
**Versión:** v1.0
**Fecha:** 2026-01-26

**Resumen del flujo:**

1. Recepción de tensor crudo `(N, 5 + C)` con formato:
   `[cx, cy, w, h, obj, p0, p1, …, pC]`
2. Normalización estructural (`_to_2d`) para garantizar matriz 2D.
3. Separación semántica de columnas:

   * coordenadas `(cx, cy, w, h)`
   * `objectness`
   * probabilidades de clase
4. Selección de clase dominante:

   * `best_cls = argmax(cls_probs)`
5. Cálculo de score final:

   * `score = objectness * max(cls_probs)`
6. Decisión de espacio de coordenadas:

   * si `runtime.out_coords_space == normalized_0_1` → escalado a píxeles del tensor
7. Salida intermedia:

   * `[cx, cy, w, h, score, class_id]`

Este flujo asume regresión directa, sin anchors ni NMS embebido.

---

### 6.2 Contrato Boxes + Scores – Flujo de desempaquetado

**ID:** FLOW-DET-BOXES-002
**Versión:** v1.0
**Fecha:** 2026-01-26

**Resumen del flujo:**

1. Recepción de dos tensores crudos:

   * uno con forma `(N, 4)` → cajas
   * otro con forma `(N, C)` → scores
2. Detección dinámica de cuál tensor representa cajas (última dimensión = 4).
3. Normalización estructural (`_to_2d`) de ambos tensores.
4. Separación semántica:

   * cajas: `(ymin, xmin, ymax, xmax)`
   * scores por clase
5. Selección de clase dominante:

   * `best_cls = argmax(scores, axis=1)`
6. Decisión de espacio de coordenadas:

   * si están normalizadas → escalado a píxeles del tensor
7. Salida intermedia:

   * `[ymin, xmin, ymax, xmax, score]`

Este contrato no incluye NMS, que se aplica posteriormente.

---

### 6.3 Contrato TFLite DetectionPostProcess – Flujo de desempaquetado

**ID:** FLOW-DET-TFLITE-003
**Versión:** v1.0
**Fecha:** 2026-01-26

**Resumen del flujo:**

1. Recepción de outputs múltiples:

   * `boxes`, `scores`, `classes`, `count` (opcional)
2. Normalización estructural de `boxes` (`_to_2d`).
3. Determinación del número efectivo de detecciones:

   * si `count` existe → `N = min(count, boxes.shape[0])`
   * si no → `N = boxes.shape[0]`
4. Recorte coherente de `boxes`, `scores` y `classes` a `N`.
5. Separación semántica de cajas:

   * `(ymin, xmin, ymax, xmax)`
6. Decisión de espacio de coordenadas:

   * si normalizadas → escalado a píxeles del tensor
7. Salida intermedia:

   * `[ymin, xmin, ymax, xmax, score]`

Este contrato ya incluye NMS y filtrado, embebidos en el grafo.

---

### 6.4 Contrato Anchor Deltas – Flujo de desempaquetado

**ID:** FLOW-DET-ANCHOR-004
**Versión:** v1.0
**Fecha:** 2026-01-26

**Resumen del flujo:**

1. Recepción de dos tensores crudos:

   * `box_deltas`
   * `class_logits`
2. Validación de dependencias externas:

   * `runtime.anchors`
   * `runtime.box_variance`
3. Detección de cuál tensor contiene deltas (última dimensión = 4).
4. Aplicación de softmax a `class_logits`.
5. Selección de clase dominante:

   * `best_cls = argmax(class_probs)`
6. Decodificación geométrica:

   * `anchor_deltas → xyxy` mediante función de decodificación
7. Normalización de cajas a `[0, 1]`.
8. Escalado a píxeles del tensor `(W, H)`.
9. Salida intermedia:

   * `[ymin, xmin, ymax, xmax, score]`

Este contrato es el más acoplado al modelo, ya que depende de anchors.

---

### 6.1 Principio

> **Cada desempaquetador conoce un contrato y solo uno.**

No existe lógica condicional genérica dentro de los unpackers.

### 6.2 Dispatch

La selección del desempaquetador se realiza mediante:

* Configuración del modelo
* Metadata del grafo
* Tipo explícito definido en JSON

Nunca por heurísticas implícitas.

---

### 7. Extensibilidad: Registry de Unpackers

Esta sección formaliza un patrón para escalar el catálogo de contratos sin crecer en complejidad accidental.

## 7.1 Patrón: UNPACKERS_REGISTRY

**ID:** EXT-PATTERN-REG-001
**Versión:** v1.0
**Fecha:** 2026-01-31

* Problema: Un if/elif grande en el selector de desempaquetadores tiende a:

crecer sin control,

mezclar lógica de selección con lógica de desempaquetado,

volver costosa la extensión (se “toca el núcleo” para agregar formatos).

Solución: usar un registry explícito (diccionario) que mapee pack_format → unpacker_fn.

* Beneficios:

agregar un contrato nuevo = escribir función + registrar (sin tocar el controlador),

facilita pruebas unitarias por contrato,

reduce ruido en unpack_out.

Esqueleto recomendado:

UNPACKERS_REGISTRY = { "yolo_flat": fn, "boxes_scores": fn, ... }

unpack_out(cfg) → registry[cfg.pack_format] con fallback seguro.

* Nota: El fallback “raw” debe seguir existiendo para depuración/control.

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

**Estado:**  descartado

**Motivo:**

* Violación de semántica
* Penalización severa de rendimiento
* Imposibilidad de inferir significado sin contexto del grafo

---

## 10. Estado actual del sistema

*  Arquitectura basada en contratos
*  Clusterización explícita
*  IR común definido
*  Formalización completa del catálogo
*  Documentación por contrato

---

## 11. Notas finales

El objetivo es preservar el criterio de diseño frente al paso del tiempo y futuras modificaciones.

---

**Fin del documento – v1.0**
