# Diseño: seam por `model_type` (estrategia-por-tarea)

**Fecha:** 2026-06-27
**Estado:** aprobado para planificar
**Relacionado:** `docs/backend-audit.md` (hallazgos H7, H13; roadmap R3, R4; precondición R1)

---

## 1. Propósito

Hoy el conocimiento de "qué tipo de modelo es y cómo se procesa de punta a punta"
está desparramado por el backend:

- `_PIPELINE_BUILDERS` decide el armado en `load_model` (rama `refactor-agente-fase1`).
- `_NEEDS_ADAPTER` es un `set` suelto en `model_controller.py` que decide si va el adapter.
- `out_coords_space` reparte el escalado entre unpacker y postproceso.
- El **contrato de salida está hardcodeado a detección**: el WS siempre responde
  `{"detections": [[x1,y1,x2,y2,conf,cls], ...], "error": null}`.

Antes de escribir la lógica real de **clasificación** y **segmentación**, hace falta
un único punto donde el `model_type` decida toda la cadena. Sin eso, agregar un tipo
nuevo obliga a tocar el WS, el controller y varios sets sueltos.

Este diseño introduce una **estrategia-por-tarea**: una estructura inmutable por
`model_type` que posee (a) cómo ensamblar los pasos del pipeline, (b) qué tipo de
resultado produce, y (c) cómo se serializa al cliente.

### Decisiones tomadas en el brainstorming

1. **Contrato de salida CLS/SEG:** no estaba decidido → se diseña acá (envelope etiquetado).
2. **Envelope WS:** etiquetado por tarea — `{task, result, error}` — el cliente hace
   `switch` por `task`. El `vision-workspace` de React ya está pensado por tipo, así que
   el costo de migrar el cliente de `detections` a `result`+`task` es chico.
3. **Alcance:** construir el seam + migrar **solo detección**; dejar CLS/SEG registradas
   pero `NotImplementedError → 501`. Sus pipelines reales se hacen después, cada uno en su
   propio PR, sin tocar el andamiaje (YAGNI: no escribir pipelines sin modelos para probarlos).

---

## 2. La abstracción central — `TaskStrategy`

Una estructura **inmutable** (no una jerarquía de clases con estado). El codebase eligió
a propósito closures/builders en vez de clases, y la reforma 8 ganó la propiedad de
**cero estado mutable por frame**. Una jerarquía con `self` mutable tienta a reintroducir
justo eso. La estrategia es solo el *bundle* que agrupa, por tipo, funciones que siguen
siendo stateless.

```python
@dataclass(frozen=True)
class TaskStrategy:
    task: str                  # "detection" | "classification" | "segmentation"
    build_pipeline: Callable   # (config, runtime) -> pasos ensamblados (closures, como hoy)
    serialize: Callable        # (resultado_crudo) -> dato JSON-listo para el envelope
```

Un registry `model_type -> TaskStrategy` **reemplaza y absorbe** el actual
`_PIPELINE_BUILDERS` de fase1: este refactor no duplica el dispatcher, lo convierte en la
versión más rica. Los pasos del pipeline siguen siendo closures stateless; el `self` del
controller no guarda nada nuevo más allá de "qué estrategia está activa".

---

## 3. Layout de archivos

```
src/api/func/tasks/
├── strategy.py        # el dataclass TaskStrategy
├── registry.py        # model_type -> TaskStrategy (reemplaza _PIPELINE_BUILDERS)
├── detection.py       # build_pipeline + serialize de deteccion
│                      #   (absorbe el _NEEDS_ADAPTER y el armado detection-actual)
├── classification.py  # registrada; build_pipeline levanta NotImplementedError -> 501
└── segmentation.py    # registrada; build_pipeline levanta NotImplementedError -> 501
```

El controller deja de contener `_PIPELINE_BUILDERS` y `_NEEDS_ADAPTER`.

---

## 4. Contrato del envelope WS (etiquetado por tarea)

Una sola forma de respuesta. El cliente hace `switch` por `task`.

```json
{ "task": "detection",
  "result": [[x1, y1, x2, y2, conf, cls], ...],
  "error": null }

{ "task": "classification",
  "result": [{"cls": 3, "score": 0.91}, ...],
  "error": null }

{ "task": "segmentation",
  "result": {"mask": "<rle|b64>", "shape": [h, w]},
  "error": null }
```

- `task` es `strategy.task` de la estrategia activa.
- `result` es lo que devuelve `strategy.serialize(...)`.
- `error` conserva el contrato actual: **el WS SIEMPRE responde** (frame inválido,
  sin modelo, fallo de inferencia) → nunca se reintroduce el deadlock del stream.

> El shape de `result` para classification/segmentation queda **especificado acá** pero
> NO se implementa en este trabajo (sus `build_pipeline` levantan 501). Es el contrato
> que sus PRs futuros deberán cumplir.

---

## 5. Flujo de datos y dónde vive la serialización

```
load_model(path):
  config   = load_model_config(path)
  strategy = registry[config.model_type]      # tipo desconocido -> 422
  pipeline = strategy.build_pipeline(...)      # NotImplementedError(CLS/SEG) -> 501
  → commit atomico: guarda (pipeline, strategy)   (si algo falla, queda descargado)

inference(img) -> resultado_crudo            # deteccion: ndarray (N,6). NO serializa.

WS /video_stream:
  out      = controller.inference(img)
  envelope = {"task":   strategy.task,
              "result": strategy.serialize(out),   # ← la estrategia serializa
              "error":  None}
```

**Punto clave:** `inference()` devuelve un **resultado de dominio** (para detección, el
ndarray `(N,6)`), y la **serialización a JSON es responsabilidad de la estrategia**, no
del controller ni del endpoint. Eso:

- hace al serializador testeable en aislamiento,
- mantiene a `inference()` agnóstico del transporte,
- y deja un único lugar por tipo donde cambia la forma del `result`.

---

## 6. Qué le saca de encima al controller (cierra H7)

- `_NEEDS_ADAPTER` (set suelto) → **se mueve adentro de `detection.py`**. El controller
  deja de saber qué es un "adapter".
- El armado detection-específico → adentro de la estrategia de detección.
- El controller queda como **manager puro**: `load` (lookup + build + commit atómico),
  `unload`, `inference` (delega al pipeline activo y devuelve el resultado de dominio),
  métricas. Nada type-específico.

**Fuera de alcance a propósito:** la deuda de "lazy scaling" / `out_coords_space` (H3,
es R5 aparte). El seam *concentra* la lógica de coordenadas dentro de la estrategia de
detección, dejándole a un futuro R5 un solo lugar donde trabajar; no la resuelve acá.

---

## 7. CLS/SEG: registradas pero 501

Sus estrategias **existen en el registry** desde el día uno, pero su `build_pipeline`
levanta `NotImplementedError` → la API responde 501 honesto (igual que hoy hace el
dispatcher de fase1). Cuando se implementen, cada una es **un archivo que se completa**
(unpacker + postproceso + serializador), sin tocar el WS, el controller ni la estrategia
de detección. Ese es el beneficio buscado: el costo marginal de un tipo nuevo es un archivo.

---

## 8. Manejo de errores en la frontera

- El contrato "el WS SIEMPRE responde" se mantiene **intacto**.
- El endpoint traduce fallos de inferencia a `{task, result: null, error: "inference_error"}`.
- **Regla del serializador:** no traga excepciones. Si algo falla serializando, propaga —
  no devuelve un envelope a medias. La frontera (WS) es la que decide el `error`.
- La taxonomía de errores tipados (R2 de la auditoría) es adyacente y queda **fuera de
  este alcance**.

---

## 9. Testing

- **Unit por estrategia:** el serializador de detección en aislamiento
  (`ndarray (N,6) → list`). Cuando lleguen CLS/SEG, cada serializador suma su unit.
- **Registry:** lookup de tipo desconocido → error; CLS/SEG → `build_pipeline` levanta 501.
- **Regresión (red de seguridad):** el e2e de YOLOv7 existente
  (`src/api/func/tests/test_end_to_end_yolov7.py`) debe seguir **verde** a través de toda
  la migración — es la garantía de que detección no cambió de comportamiento.
- **Envelope:** un test del endpoint/WS que verifica la forma `{task, result, error}` para
  detección (incluyendo los caminos de error: sin modelo, frame inválido).

---

## 10. Precondición

Este trabajo se apoya en la **base mergeada** (R1 de la auditoría:
`refactor-agente-fase1` ↔ `refactor-frontend-react`), porque parte de la madurez del
controller de fase1 (`_stats_lock`, lock acotado para concurrencia, y el dispatcher
`_PIPELINE_BUILDERS` que este diseño absorbe). **El merge es el bloqueante previo** y no
está cubierto por este spec.

---

## 11. Fuera de alcance (resumen explícito)

- Implementar los pipelines reales de clasificación y segmentación (solo se especifica su
  contrato de `result`).
- Resolver la deuda de lazy scaling / `out_coords_space` (R5).
- La taxonomía de errores tipados de dominio (R2).
- El merge de ramas (R1, precondición).
- Cambios en el cliente React más allá de migrar el lector del envelope de `detections`
  a `result` + `task` (la implementación de las vistas CLS/SEG va con sus PRs).
