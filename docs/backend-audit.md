# UNCaLens — Auditoría del Backend

> Documento de auditoría objetiva del backend (FastAPI + pipeline de inferencia
> multi-backend). Equivalente, para el backend, a lo que `docs/frontend-components/`
> es para el frontend: una foto de **cómo está** y una guía de **cómo debería estar**,
> con un roadmap priorizado.
>
> **Fecha:** 2026-06-27
> **Base auditada:** rama `refactor-frontend-react` (endpoints thin-client) **+** la
> Fase 2 de `refactor-agente-fase1` tomada como estado objetivo combinado. Es decir:
> se audita el backend *post-merge esperado*, no el de una sola rama. Ver §6 para por
> qué ese merge todavía no existe y qué implica.

---

## 1. Propósito y alcance

Este doc cubre **todo el backend Python**: la API (`src/api/mainAPI.py`), el orquestador
(`model_controller.py`), los tres pipelines (`reader_pipeline` / `input_pipeline` /
`output_pipeline`), los loaders por framework (`forms/`), los unpackers, el schema
(`config_schema.py`) y el logger. No cubre el frontend React ni el contenedor Electron
(ya documentados en `docs/frontend-components/`).

**Advertencia de partida (drift):** el `CLAUDE.md` describe el backend con la Fase 2 ya
integrada, pero esa Fase 2 vive en `refactor-agente-fase1` y **no está mergeada** en la
rama de trabajo actual. Varias afirmaciones de la guía no se corresponden con el código
de ninguna rama tomada por separado. El detalle puntual está en §6; el plan de merge es
el hallazgo R1 del roadmap (§5).

**Taxonomía de severidad usada en §3:**

- 🔴 **Corrección / concurrencia** — puede producir resultados incorrectos, crashes o
  data races.
- 🟠 **Deuda de diseño** — funciona, pero limita la extensibilidad o el mantenimiento.
- 🟡 **Menor** — wart localizado, bajo impacto.

---

## 2. Arquitectura actual ("cómo está")

### 2.1 El flujo central

```
frame JPEG ─WS─> mainAPI ─> ModelController.inference(img_rgb)
  1. preprocess_fn(img) -> (tensor, meta)     letterbox/resize + escala/normalización
  2. input_adapter(tensor)                     color_order + layout + dtype
  3. predict_fn(x)                             inferencia del backend (forms/*)
  4. unpack_fn(raw, runtime) -> (N,K) float32  tensor crudo -> matriz 2D
  5. output_adapter(row)                       reordena a [x1,y1,x2,y2,conf,cls]
  6. postprocess_fn(arr, meta)                 conf filter -> top-k -> NMS -> undo letterbox
<─WS─ JSON {detections: [[x1,y1,x2,y2,conf,cls], ...], error}
```

Cada paso es un **closure construido por un builder** (`build_preprocessor`,
`generate_input_adapter`, `unpack_out`, `buildPostprocessor`, …). El controller los
arma una vez en `load_model` y los compone en `inference`.

### 2.2 Lo que está objetivamente bien

Esto no es relleno: el backend tiene decisiones de diseño sólidas que conviene **no
romper** en las reformas.

- **Separación de pipelines limpia.** `reader` / `input` / `output` con responsabilidades
  disjuntas. Cada paso es una función casi pura, testeable en aislamiento. El patrón
  builder/closure es consistente en todo el código.
- **Agnóstico al framework de verdad.** El JSON + schema Pydantic estricto
  (`extra="forbid"`, `config_schema.py:10`) es un contrato fuerte: un campo mal escrito
  es un error de carga visible, no un silencio.
- **`load_model` atómico con rollback** (`model_controller.py`): el estado del controller
  solo se commitea si *todo* el armado salió bien; ante cualquier fallo queda descargado
  y la excepción se propaga. La API la mapea a 404/422/501/500 honestos.
- **`validate_pipeline()`**: corre una inferencia dummy end-to-end tras cargar, así el
  desajuste JSON↔modelo se detecta al cargar y no en pleno stream.
- **Estado por-frame en `meta`** (reforma 8): `preprocess_fn → (tensor, meta)` y
  `postprocess_fn(arr, meta)`. No hay estado mutable compartido entre frames.
- **Concurrencia real (Fase 2, en fase1):** el `_lock` solo cubre el snapshot atómico
  del pipeline; el trabajo pesado (pre/predict/post) corre sin el lock. `_stats_lock`
  separado para diagnóstico. TFLite se serializa con su propio lock interno
  (`tflite_load.py:103`) porque su `Interpreter` no es thread-safe.
- **Contrato único de shape (Fase 2):** `normalize_to_2d` (`unpackers/_shape.py`)
  garantiza que todo unpacker entrega `(N,K)` float32 2D; el controller ya no normaliza.
- **Dispatcher honesto (Fase 2):** `_PIPELINE_BUILDERS` enruta por `model_type`;
  `classification`/`segmentation` están reconocidos pero levantan `NotImplementedError`
  con un checklist detallado → la API responde **501 honesto** en vez de fingir soporte.
- **Hot-path consciente:** normalización + escala fusionadas en una pasada numpy
  (`input_transformer.py:139`); `predict_fn` devuelve numpy, nunca `.tolist()`; frames
  binarios en el WS; inferencia en threadpool para no congelar el event loop.
- **El WS siempre responde** (incluso ante frame inválido / sin modelo / fallo): nunca
  deja al cliente esperando un frame que no llega.

---

## 3. Hallazgos ("cómo está", los problemas)

| # | Sev | Hallazgo | Ubicación |
|---|-----|----------|-----------|
| H1 | 🔴 | **Divergencia de ramas conflictiva.** `mainAPI.py` divergió: fase1 tiene `/model/output_shape` + `/metrics/snapshot`; frontend-react tiene `/models`, `GET /configs/{name}`, `/models/upload`. Ninguna es superset. El `CLAUDE.md` describe una unión que no existe en ninguna rama. | `mainAPI.py` (ambas ramas) |
| H2 | 🟠 | **`inference()` es detection-shaped.** El load-time despacha por `model_type`, pero el run-time hardcodea la salida de detección (`output_adapter` + postproceso de cajas). CLS/SEG van a necesitar inferencia despachada por tipo, o que cada pipeline traiga su propia `inference_fn`. La abstracción está hecha a la mitad. | `model_controller.py` `inference()` |
| H3 | 🟠 | **Deuda de "lazy scaling".** El ownership del espacio de coordenadas (normalizado vs. píxeles del tensor) está repartido entre unpacker, `output_adapter` y postprocesador. Admitido en el propio código. | `unpackers/utils.py:87`, `output_transformer.py:_undo_transform_xyxy_inplace` |
| H4 | 🔴 | **`except Exception as e: raise ValueError(e)`** destruye el tipo original y el traceback. Convierte cualquier fallo (incl. errores de programación) en un `ValueError` opaco que la API mapea a 422 como si fuera culpa del usuario. En **ambas** ramas. | `input_transformer.py:164` |
| H5 | 🟠 | **Errores de stream efímeros y opacos.** Los fallos de inferencia van solo a un `deque` in-memory (se pierden al reiniciar) y al cliente le llega un genérico `"inference_error"` sin id de correlación ni categoría. | `mainAPI.py` (`_inference_errors`, WS) |
| H6 | 🟡 | **Ramas de config muertas en los loaders.** `keras_load.py:81` (`runtime_cfg.tf`) y comentarios admiten configs que no existen en el schema → código que nunca se ejecuta. | `keras_load.py`, `tflite_load.py` |
| H7 | 🟠 | **El controller sabe de más.** `_NEEDS_ADAPTER`, generación de anchors y validación de índices viven en el controller. El propio TODO lo admite. No es un "administrador de pipelines" puro. | `model_controller.py` |
| H8 | 🟡 | **Loop Python por detección en el hot path.** `[output_adapter(r) for r in unpacked]` recorre N detecciones en Python puro; el `output_adapter` es mapeo de columnas, vectorizable con numpy. | `model_controller.py` `inference()` |
| H9 | 🟠 | **Techos estructurales.** Controller global único, un modelo cargado, un umbral, un "usuario". Para una app de escritorio mono-usuario alcanza, pero es un límite duro a documentar explícitamente. | `mainAPI.py:49` `controller = ModelController()` |
| H10 | 🟠 | **PyTorch = solo TorchScript.** `MODEL_EXTENSIONS` anuncia `.pt`/`.pth`, pero `pytorch_load.py` solo carga modelos `torch.jit`. Un `.pth` de `state_dict` (el caso más común) falla con un error que no explica *por qué* hace falta exportar a JIT. | `pytorch_load.py:51` |
| H11 | 🟡 | **`CORS allow_origins=["*"]` + endpoints de escritura sin auth.** Para una app localhost mono-usuario el riesgo es bajo, pero `POST /configs/{name}`, `/models/upload` y `/select_model` escriben/leen disco y cualquier página local podría invocarlos. | `mainAPI.py:42` |
| H12 | 🟡 | **Warts del schema.** `top_k: int = False` usa bool como default de int (`config_schema.py:81`); el nombre del logger sale de `basename.split(".")[0]`, que trunca nombres de archivo con puntos. | `config_schema.py`, `model_controller.py` |
| H13 | 🟠 | **CLS/SEG: superficie de schema sin runtime.** `ClassificationOutput` y `SemanticSegmentationOutput` están completos en el schema (con `label_map`, `colormap`, `output_stride`…) pero sin pipeline ni tests. El schema promete más de lo que el sistema entrega (hoy mitigado por el 501 honesto). | `config_schema.py:88-101` |
| H14 | 🟡 | **La concurrencia (el corazón de Fase 2) no está load-tested.** Hay `test_controller_phase2.py`, pero ningún test prueba que dos inferencias concurrentes reales no corrompan estado compartido bajo carga. | `tests/` |

---

## 4. Estado objetivo ("cómo debería estar")

### 4.1 Reconciliación de ramas (resuelve H1)

El backend objetivo es **un solo `mainAPI.py`** que une los dos conjuntos de endpoints:

| Endpoint | Origen | Estado objetivo |
|---|---|---|
| `/get_models`, `/select_model`, `/model/load`, `/model/unload`, `/config/confidence`, `/config/template/{type}`, `POST /configs/{name}`, `/video_stream`, `/logs/inference`, `/metrics` | común | conservar |
| `/models`, `GET /configs/{name}`, `/models/upload` | frontend-react | **traer a fase1** (thin client) |
| `/model/output_shape/{name}`, `/metrics/snapshot` | fase1 | **traer a frontend-react** |

El controller objetivo es el de fase1 (concurrente, dispatcher, `_shape.py`). El
`CLAUDE.md` debe reescribirse para describir ese estado unificado (§6).

### 4.2 Inferencia despachada por tipo (resuelve H2)

Cada builder de `_PIPELINE_BUILDERS` debería devolver, además de los pasos, una
**`inference_fn(snapshot, img) -> resultado`** propia del tipo (o el controller debería
despachar el lado de salida por `model_type`). Así `inference()` deja de asumir cajas y
CLS/SEG pueden devolver su propio contrato (vector de clases / máscara) sin parchear el
camino de detección. El lado de **input** (`predict_fn`/`preprocess`/`input_adapter`) ya
es genérico y se reusa tal cual.

### 4.3 Un único dueño del espacio de coordenadas (resuelve H3)

Definir explícitamente **qué capa** posee la conversión normalizado↔píxeles. Propuesta:
el unpacker entrega *siempre* píxeles del tensor (contrato declarado), y el postprocesador
solo deshace letterbox/resize. Eliminar `out_coords_space` como bandera repartida y
documentar el contrato en un solo lugar.

### 4.4 Taxonomía de errores (resuelve H4, H5, H6)

Aplicando patrones de manejo de errores: reemplazar el catch-all por **excepciones de
dominio tipadas** que el límite (API/WS) traduce, en vez de que cada capa decida el
código HTTP:

```
ConfigError(ValueError)        -> 422   (JSON inválido, índices fuera de rango)
ModelLoadError(RuntimeError)   -> 404/500 (archivo, backend, pesos)
UnsupportedModelType(...)      -> 501   (CLS/SEG no implementados)
InferenceError(RuntimeError)   -> WS error con categoría + id de correlación
```

Reglas concretas derivadas:
- **Nunca** `except Exception: raise OtroTipo(e)` sin `from e`. El `input_transformer`
  debe dejar propagar el error real o envolverlo con `raise ... from e` preservando tipo
  y causa.
- **Falla rápido y específico** en la frontera (load), no en mitad del hot path.
- **Errores de stream persistentes y categorizados:** además del deque, loguear al
  archivo del modelo con categoría; devolver al cliente `{error: {code, detail, id}}`
  para que el frontend pueda distinguir "frame inválido" de "modelo incompatible".
- **Borrar las ramas de config muertas** (`runtime_cfg.tf`, etc.): código que no se
  ejecuta es deuda que confunde.

### 4.5 Controller como administrador puro (resuelve H7, H8)

Mover `_NEEDS_ADAPTER`, la generación de anchors y la validación de índices **dentro de
los builders/unpackers**, donde es conocimiento local del pipeline de detección. El
controller queda como: snapshot atómico → despacho → métricas. Vectorizar el
`output_adapter` para que opere sobre la matriz `(N,K)` completa en vez de fila por fila.

### 4.6 Límites estructurales explícitos (resuelve H9, H10, H11)

- Documentar que el sistema es **mono-modelo / mono-usuario** por diseño (no es un bug,
  es un alcance). Si en el futuro se quiere multi-modelo, el camino es un registro de
  controllers por sesión, no parchear el global.
- **PyTorch:** o se soporta `state_dict` (requiere la clase `nn.Module`, lo cual rompe el
  modelo "agnóstico por JSON" — ver la discusión de grafos) o se documenta claramente que
  solo se aceptan `.pt`/`.pth` exportados con `torch.jit.save()`, con un mensaje de error
  que lo diga al cargar. Recomendado: lo segundo (mantiene la pureza del contrato).
- **CORS/escritura:** restringir `allow_origins` al origen real de Electron y/o exigir un
  token local en los endpoints de escritura. Bajo riesgo, arreglo barato.

### 4.7 CLS/SEG: implementar o congelar (resuelve H13)

El andamiaje del schema es valioso pero hoy es superficie sin runtime. Decisión de
diseño pendiente (requiere modelos reales para validar): implementar los builders +
unpackers + el **contrato de salida con el cliente** (un vector de clases y una máscara
HxW no caben en el JSON de cajas actual), o marcar el schema CLS/SEG como experimental
hasta tener modelos.

---

## 5. Roadmap priorizado

Reformas numeradas, alta → baja. Esfuerzo: S (horas) / M (1-2 días) / L (varios días).

| ID | Prioridad | Reforma | Resuelve | Esfuerzo | Depende de |
|----|-----------|---------|----------|----------|-----------|
| **R1** | 🔴 Alta | **Merge fase1 ↔ frontend-react.** Unificar `mainAPI.py` (los dos sets de endpoints), tomar el controller de fase1, correr los tests de ambas ramas, reescribir `CLAUDE.md` al estado unificado. **Bloqueante de todo lo demás.** | H1 | — |
| **R2** | 🔴 Alta | **Taxonomía de errores.** Excepciones de dominio tipadas + traducción en la frontera; arreglar `input_transformer` (`from e`); persistir y categorizar errores de stream; borrar ramas de config muertas. | H4, H5, H6 | R1 |
| **R3** | 🟠 Media | **Inferencia despachada por tipo.** Cada pipeline trae su `inference_fn`; `inference()` deja de asumir cajas. Desbloquea CLS/SEG sin parches. | H2 | R1 |
| **R4** | 🟠 Media | **Controller como administrador puro** + vectorizar `output_adapter`. Mover lógica de detección a sus builders/unpackers. | H7, H8 | R1, R3 |
| **R5** | 🟠 Media | **Un dueño del espacio de coordenadas.** Unificar la conversión normalizado↔píxeles en una sola capa; eliminar la bandera repartida. | H3 | R4 |
| **R6** | 🟠 Media | **Límites estructurales explícitos.** Mensaje claro de PyTorch=TorchScript; CORS acotado + token en endpoints de escritura; documentar mono-modelo/mono-usuario. | H9, H10, H11 | R1 |
| **R7** | 🟡 Baja | **Warts del schema.** `top_k: int = 0`; nombre de logger robusto a puntos en el filename. | H12 | R1 |
| **R8** | 🟠 Media | **Test de concurrencia real.** Cargar un modelo y disparar N inferencias concurrentes verificando que no se corrompe el estado ni las métricas. | H14 | R1 |
| **R9** | 🟠 Media (decisión) | **CLS/SEG: implementar o congelar.** Requiere modelos reales + decisión del contrato de salida con el cliente. | H13, H2 | R3 |

**Orden sugerido:** R1 primero (sin esto todo lo demás se hace dos veces). Después R2
(corrección, barato y de alto impacto). Luego R3→R4→R5 como una cadena de
re-arquitectura del lado de salida. R6/R7/R8 en paralelo cuando convenga. R9 cuando haya
modelos CLS/SEG para validar.

---

## 6. Apéndice: drift con CLAUDE.md

Afirmaciones del `CLAUDE.md` que **no se corresponden con la rama de trabajo
`refactor-frontend-react`** (sí están en `refactor-agente-fase1`, sin mergear):

- "El `RLock` ahora protege SOLO load/unload y el snapshot atómico; `inference()` NO lo
  sostiene" → en `refactor-frontend-react`, `inference()` **sí** toma el `RLock` durante
  todo el método: los streams se serializan por completo.
- "`_stats_lock` aparte" → no existe en esta rama.
- "`normalize_to_2d` en `unpackers/_shape.py`" → el archivo no existe en esta rama; la
  normalización de shape sigue **duplicada** en `inference()` + `raw.py`.
- "Despacho por `model_type` vía `_PIPELINE_BUILDERS`" → en esta rama sigue el viejo
  `if config.model_type != "detection": raise`.
- "TFLite se serializa con su propio lock interno" → en esta rama `tflite_load.py` no
  tiene lock (y como acá la inferencia está serializada por el `RLock` global, no se
  manifiesta — pero la afirmación es falsa para esta rama).

Afirmaciones ya marcadas como no-implementadas por el propio `CLAUDE.md` y confirmadas:
`GET /model/output_shape/{model_name}` y `POST /metrics/snapshot` **sí** existen en
`refactor-agente-fase1` pero **no** en `refactor-frontend-react`.

**Acción (parte de R1):** tras el merge, reescribir el `CLAUDE.md` para que describa un
único estado real y verificable, y mantener este `backend-audit.md` como el registro de
deuda viva.
