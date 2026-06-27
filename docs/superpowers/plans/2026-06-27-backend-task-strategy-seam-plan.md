# Plan de implementación: seam por `model_type` (estrategia-por-tarea)

**Fecha:** 2026-06-27
**Spec:** `docs/superpowers/specs/2026-06-27-backend-task-strategy-seam-design.md`
**Auditoría:** `docs/backend-audit.md` (cierra H7; avanza R3/R4; precondición R1)

---

## Precondición (NO cubierta por este plan)

**P0 — base mergeada (R1):** `refactor-agente-fase1` ↔ `refactor-frontend-react`.
El plan asume la madurez del controller de fase1 (`_stats_lock`, lock acotado,
`_PIPELINE_BUILDERS`). Este último **se absorbe** en el registry de estrategias (Fase A),
no se mantiene en paralelo.

> Estado actual verificado: la rama `refactor-frontend-react` todavía tiene el controller
> viejo (`if config.model_type != "detection": raise` en `model_controller.py:74`,
> `_NEEDS_ADAPTER` en `:28`). El plan describe el estado objetivo post-merge.

---

## Principios de error-handling aplicados

De `error-handling-patterns`, las reglas que se hornean en cada paso:

- **Errores tipados traducidos en la frontera:** excepciones de dominio
  (`UnknownModelType`, `TaskNotImplemented`) que la API mapea a 422/501. No strings
  genéricos, no `except Exception` que disfraza bugs (eso es H4).
- **No tragar:** el serializador propaga; nunca devuelve un envelope a medias.
- **Manejar en el nivel correcto:** la traducción a HTTP/envelope vive en la frontera
  (endpoints WS/REST), no en el medio del pipeline.
- **El WS SIEMPRE responde:** se preserva intacto el contrato anti-deadlock.

---

## Fase A — Backend: construir el seam (sin cambio de comportamiento para detección)

### A1 — Excepciones tipadas de la frontera
**Archivo nuevo:** `src/api/func/tasks/errors.py`
- `class UnknownModelType(ValueError)` — `model_type` no registrado. → API 422.
- `class TaskNotImplemented(NotImplementedError)` — tarea reconocida pero sin pipeline. → API 501.
- Docstrings en español (convención del repo).

**Verificación:** import limpio; `pytest` no rompe nada todavía.

### A2 — El dataclass `TaskStrategy`
**Archivo nuevo:** `src/api/func/tasks/strategy.py`
```python
@dataclass(frozen=True)
class TaskStrategy:
    task: str                  # "detection" | "classification" | "segmentation"
    build_pipeline: Callable   # (config, runtime) -> pasos ensamblados (closures)
    serialize: Callable        # (resultado_crudo) -> dato JSON-listo
```
Inmutable a propósito (preserva reforma 8: cero estado mutable por frame).

**Verificación:** instanciable; `frozen=True` impide reasignación (test trivial).

### A3 — Estrategia de detección (absorbe la lógica actual)
**Archivo nuevo:** `src/api/func/tasks/detection.py`
- `build_pipeline(config, runtime)`: mueve acá el armado detection-específico del
  controller **y la decisión `_NEEDS_ADAPTER`** (el `if pack_fmt in _NEEDS_ADAPTER` de
  `model_controller.py:199` pasa a vivir dentro de esta función). El controller deja de
  conocer "adapter".
- `serialize(arr) -> list`: `arr.tolist()` de la matriz `(N,6)`. Es serialización **en
  la respuesta** (no en el hot path de `predict_fn`), igual costo que el `.tolist()` que
  hoy hace `mainAPI` al armar el JSON. **No traga:** si `arr` no es la matriz esperada,
  propaga.

**Verificación:** unit del serializador en aislamiento (A-test en Fase C).

### A4 — Estrategias CLS/SEG registradas pero 501
**Archivos nuevos:** `src/api/func/tasks/classification.py`, `segmentation.py`
- `build_pipeline(...)` levanta `TaskNotImplemented` con mensaje honesto.
- `serialize(...)` placeholder que también levanta `TaskNotImplemented` (nunca alcanzado;
  el 501 corta antes en load).
- El contrato de `result` futuro (top-k / máscara) queda documentado en el docstring,
  citando el spec §4.

**Verificación:** import limpio; el build levanta la excepción tipada.

### A5 — Registry por tipo
**Archivo nuevo:** `src/api/func/tasks/registry.py`
- `TASK_STRATEGIES: dict[str, TaskStrategy]` con las tres entradas.
- `get_strategy(model_type) -> TaskStrategy`: levanta `UnknownModelType` si falta.
- **Reemplaza** `_PIPELINE_BUILDERS`.

**Verificación:** `get_strategy("detection")` ok; `get_strategy("foo")` → `UnknownModelType`.

### A6 — Cablear el controller como manager puro
**Archivo:** `src/api/func/model_controller.py`
- En `load_model`: reemplazar el `if model_type != "detection"` y/o `_PIPELINE_BUILDERS`
  por `strategy = get_strategy(config.model_type)` + `pipeline = strategy.build_pipeline(...)`.
- Guardar `self._strategy` en el **commit atómico** (junto al pipeline; si algo falla,
  queda descargado y la excepción tipada se propaga — sin cambios al patrón atómico).
- Borrar `_NEEDS_ADAPTER` del controller (migrado a A3).
- `inference()`: **sin cambios de retorno** — devuelve el resultado de dominio (ndarray
  `(N,6)` para detección). Conserva la validación de índices de `tensor_structure`.
- Exponer la estrategia activa para la frontera: `controller.active_task` (str) y
  `controller.serialize_result(out)` (delega a `self._strategy.serialize`), o exponer
  `self._strategy` de solo lectura. Elegir la que deje la frontera más limpia.

**Verificación:** `pytest` unitarios verdes (sin tocar comportamiento de detección).

### A7 — Frontera: endpoint WS arma el envelope etiquetado
**Archivo:** `src/api/mainAPI.py`
- En `/video_stream`, construir:
  ```python
  envelope = {"task": controller.active_task,
              "result": controller.serialize_result(out),
              "error": None}
  ```
- **Traducción de errores en la frontera (no en el medio):**
  - sin modelo → `{"task": None, "result": None, "error": "no_model"}` (igual que hoy).
  - frame inválido → `error: "frame_invalido"`.
  - fallo de inferencia → `try/except` que captura la excepción de inferencia, la loguea
    (deque `_inference_errors`) y responde `error: "inference_error"`. **El WS SIEMPRE
    responde** — no se reintroduce el deadlock.
- En `/select_model` y `/model/load`: mapear `UnknownModelType` → `HTTPException(422)` y
  `TaskNotImplemented` → `HTTPException(501)`. Reemplaza el manejo ad-hoc actual.

**Verificación:** test de endpoint (Fase C): forma del envelope + caminos de error.

---

## Fase B — Cliente React: migrar el lector del envelope (mínimo)

### B1 — `detection.service.ts` lee `payload.result`
**Archivo:** `client/src/features/vision-workspace/services/detection.service.ts`
- `parse(payload)`: leer `payload.result` (matriz `(N,6)`) en vez de `payload.detections`.
- Guard opcional de consistencia: si `payload.task` existe y `!== 'detection'`, devolver `null`.
- `present.ts` **no se toca**: ya despacha por `modelType` conocido y ya maneja
  `payload.error` correctamente (líneas 38-43). El `task` del envelope es señal de
  consistencia, no driver del dispatch (el cliente ya sabe el tipo del modelo cargado).

**Verificación:** `npm run typecheck` (en `client/`) verde.

> CLS/SEG del cliente (`classification.service.ts`, `segmentation.service.ts`) ya son
> stubs con `implemented: false` y TODO del contrato — se completan con sus PRs, fuera
> de este alcance.

---

## Fase C — Tests

### C1 — Unit del serializador de detección
`ndarray (N,6)` → `list` de listas; matriz vacía → `[]`; entrada malformada → propaga.

### C2 — Registry
- `get_strategy("detection")` devuelve la estrategia.
- `get_strategy("desconocido")` → `UnknownModelType`.
- `classification`/`segmentation` `build_pipeline(...)` → `TaskNotImplemented`.

### C3 — Endpoint / envelope
- Detección: respuesta WS con forma `{task: "detection", result: [...], error: None}`.
- Caminos de error: sin modelo (`no_model`), frame inválido (`frame_invalido`),
  fallo de inferencia (`inference_error`) — todos responden (nunca cuelga).
- `/select_model` sobre un modelo CLS → 501; tipo desconocido → 422.

### C4 — Regresión (red de seguridad)
- `src/api/func/tests/test_end_to_end_yolov7.py` **sigue verde**: garantía de que
  detección no cambió de comportamiento end-to-end.

### C5 — Cliente
- `npm run typecheck` y `npm run build` en `client/`.

---

## Orden de ejecución y dependencias

```
P0 (merge, externo) ──> A1 ──> A2 ──> A3 ──> A4 ──> A5 ──> A6 ──> A7 ──> C1..C4
                                                                  └─> B1 ──> C5
```

- A1→A5 son aditivos (archivos nuevos), no rompen nada hasta A6.
- A6 es el corte: el controller pasa a usar el registry. A partir de acá, `pytest` valida.
- A7 cambia el contrato WS → **obliga** a B1 en el mismo PR (si no, el cliente de
  detección deja de dibujar).
- C4 es la condición de "no regresión" innegociable.

## Criterios de aceptación (Definition of Done)

1. `pytest` verde (incluye e2e de YOLOv7 sin cambios de comportamiento).
2. `npm run typecheck` + `npm run build` verdes en `client/`.
3. Cargar un modelo de detección → el WS responde `{task, result, error}` y el cliente
   dibuja igual que antes.
4. Cargar un modelo CLS/SEG → 501 honesto (sin tocar el WS ni el controller).
5. `model_controller.py` ya no contiene `_NEEDS_ADAPTER` ni `_PIPELINE_BUILDERS`; la
   decisión de adapter vive en `tasks/detection.py`.
6. Un `model_type` desconocido → 422 con excepción tipada, no `except Exception`.

## Fuera de alcance (recordatorio del spec §11)

- Pipelines reales de CLS/SEG (solo su contrato de `result`).
- Deuda de lazy scaling / `out_coords_space` (R5).
- Taxonomía completa de errores de dominio (R2).
- El merge de ramas (P0/R1).
- Limpieza del frontend vanilla (`src/render/` + `static/`).
