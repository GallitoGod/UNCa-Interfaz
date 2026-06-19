# Especificación de Requisitos de Software (ERS)

## UNCaLens — Interfaz de inferencia de modelos de visión por computadora

| Campo | Valor |
|---|---|
| Proyecto | UNCaLens (UNCa-Interfaz) |
| Versión del documento | 1.1 |
| Fecha | 2026-06-12 |
| Estado del sistema descrito | Estado actual del repositorio (rama `main`, post-reformas 6 y 8 del 2026-06-12) |
| Estándar de referencia | IEEE 830-1998 (adaptado) |

---

## 1. Introducción

### 1.1 Propósito

Este documento especifica los requisitos funcionales y no funcionales de **UNCaLens**,
una aplicación de escritorio para ejecutar modelos de detección de objetos sobre
cámara en vivo, archivos de video e imágenes. Describe el sistema **tal como está
implementado a la fecha**, incluyendo sus límites actuales, para servir de referencia
a desarrolladores, evaluadores y futuros mantenedores.

### 1.2 Alcance

UNCaLens permite a un usuario:

- Cargar modelos de detección de objetos de distintos frameworks (ONNX Runtime,
  TensorFlow Lite, Keras/TensorFlow, PyTorch/TorchScript) descritos mediante un
  archivo JSON de configuración por modelo.
- Ejecutar inferencia en tiempo real sobre webcam, o sobre videos e imágenes cargados
  desde archivo.
- Visualizar las detecciones (cajas, confianza, clase) superpuestas sobre la imagen,
  ajustar el umbral de confianza en vivo y grabar la salida.
- Crear configuraciones de modelos nuevos mediante un asistente (wizard) de 4 pasos.
- Consultar métricas de rendimiento y errores de inferencia.

**Fuera del alcance actual** (el backend los rechaza honestamente con HTTP 501):
pipelines de clasificación y segmentación. El esquema de configuración ya los
contempla, pero el controlador solo arma pipelines de detección.

### 1.3 Definiciones, acrónimos y abreviaturas

| Término | Definición |
|---|---|
| ERS | Especificación de Requisitos de Software |
| WS | WebSocket |
| Backend de inferencia | Runtime que ejecuta el modelo: `onnxruntime`, `tflite`, `tensorflow` (Keras), `pytorch` |
| Config / JSON de modelo | Archivo `configs/<nombre>.json` que describe preprocesado, desempaquetado de salida y runtime de un modelo |
| Unpacker | Componente que convierte el tensor crudo de salida del modelo en una matriz estándar (N,6) |
| `pack_format` | Identificador del unpacker a usar: `raw`, `yolo_flat`, `boxes_scores`, `tflite_detpost`, `anchor_deltas` |
| Letterbox | Redimensionado que preserva relación de aspecto rellenando con un color de padding |
| NMS | Non-Maximum Suppression |
| Formato estándar interno | Detección como `[x1, y1, x2, y2, conf, cls]` en píxeles de la imagen original |
| Plantilla | JSON de ejemplo en `configs/plantillas/`; no se lista como modelo |

### 1.4 Referencias

- `README.md`, `requirements.txt`.
- Documentación de FastAPI, Electron, ONNX Runtime, TensorFlow Lite, PyTorch.

### 1.5 Visión general del documento

La sección 2 describe el producto en general (perspectiva, funciones, restricciones).
La sección 3 detalla las interfaces externas. La sección 4 enumera los requisitos
funcionales y la sección 5 los no funcionales. La sección 6 documenta limitaciones
conocidas y deuda técnica vigente.

---

## 2. Descripción general

### 2.1 Perspectiva del producto

Sistema de dos procesos que corren en la misma máquina:

```
┌─────────────────────────────┐         HTTP / WebSocket          ┌──────────────────────────────┐
│  Frontend — Electron        │  ───────────────────────────────▶ │  Backend — FastAPI (Python)  │
│  (captura, UI, dibujo,      │  frames JPEG binarios (WS out)    │  (pipeline de inferencia,    │
│   wizard de configs)        │ ◀───────────────────────────────  │   gestión de modelos)        │
│                             │  JSON {detections, error} (WS in) │  127.0.0.1:8000              │
└─────────────────────────────┘                                   └──────────────────────────────┘
```

- **Backend**: Python 3.8.10, FastAPI + WebSocket (`src/api/`). Recibe frames JPEG
  binarios y responde **JSON con detecciones**; nunca devuelve imágenes renderizadas.
- **Frontend**: Electron 32 (`src/render/`, proceso principal en `src/main.js`).
  Captura webcam/archivo, envía frames por WS, y **dibuja localmente** el frame más
  las cajas en un canvas (`overlay.js`). El renderer corre **aislado**
  (`contextIsolation` + `sandbox`, sin Node): el acceso a disco vive en el proceso
  principal detrás de IPC (`src/preload.js` expone `window.uncaAPI`;
  `src/ipc-handlers.js` valida y ejecuta).
- Los procesos se arrancan por separado (uvicorn + `npm start`); no hay spawn
  automático del backend desde Electron (mejora pendiente).

### 2.2 Funciones del producto (resumen)

1. **Gestión de modelos**: listar modelos disponibles (config + pesos presentes),
   cargar, descargar, importar pesos por dropzone.
2. **Inferencia en vivo**: streaming de frames por WS con control de flujo
   (1 frame en vuelo), sobre webcam con selección de cámara y espejo, o sobre
   video/imagen de archivo.
3. **Visualización client-side**: dibujo de cajas y etiquetas con colores
   configurables localmente; grabación del canvas de salida (MediaRecorder).
4. **Configuración de modelos**: wizard de 4 pasos que genera `configs/<modelo>.json`
   validado contra un esquema Pydantic estricto al momento de la carga.
5. **Ajuste en vivo**: umbral de confianza modificable durante el streaming.
6. **Observabilidad**: métricas de rendimiento (latencias, FPS), últimos errores de
   inferencia, logs rotativos por modelo.

### 2.3 Características de los usuarios

- **Usuario final / investigador**: opera la UI para probar modelos sobre cámara o
  archivos. No necesita conocimientos de programación.
- **Integrador de modelos**: usuario técnico que agrega un modelo nuevo: coloca los
  pesos en `models/` y crea el JSON con el wizard o a mano. Debe entender el contrato
  de entrada/salida de su modelo (layout, formato de cajas, espacio de coordenadas).

### 2.4 Restricciones generales

| Id | Restricción |
|---|---|
| RG-1 | El backend requiere **Python 3.8.10** (entorno congelado; las dependencias no están pineadas — ver §6). |
| RG-2 | Backend y frontend corren en `localhost`; el puerto **8000** está hardcodeado en `src/render/modules/constants.js`. |
| RG-3 | El renderer de Electron corre con la configuración segura recomendada (`contextIsolation: true`, `sandbox`, sin `nodeIntegration`): solo puede tocar el disco a través de los 4 canales IPC validados del proceso principal (listar/importar modelos, leer/escribir configs). |
| RG-4 | El servicio atiende **un stream de video a la vez**: el estado mutable compartido por frame fue eliminado (el metadata viaja con cada frame), pero el `RLock` del controller todavía serializa las inferencias completas (reforma 8b pendiente para concurrencia real). |
| RG-5 | Solo se soporta `model_type: "detection"`; clasificación y segmentación se rechazan con 501. |
| RG-6 | GPU opcional: `onnxruntime-gpu` fijado a 1.17.1 (CUDA 11.8). |

### 2.5 Suposiciones y dependencias

- Existe al menos una webcam o archivos de prueba para usar el streaming.
- Los pesos de los modelos están en `models/` con extensión soportada
  (`.onnx`, `.tflite`, `.h5`, `.keras`, `.pt`, `.pth`).
- El nombre base del JSON de config coincide con el nombre base del archivo de pesos
  (`_find_model_file` resuelve la extensión de forma determinista por preferencia).
- Modelos incluidos hoy en el repo: `yolov7-tiny.onnx`, `efficientdet-lite0.tflite`,
  `efficientdet-lite2.tflite` (con config).

---

## 3. Interfaces externas

### 3.1 Interfaz de usuario (Electron)

- **Vista principal**: video de entrada (cámara o archivo) + canvas de salida con
  detecciones superpuestas; selector de modelo, selector de cámara, slider de
  confianza, control de grabación.
- **Vista "Modelos"**: lista de pesos en `models/` (escaneo vía IPC al proceso
  principal), dropzone para importar archivos, acceso al wizard de configuración.
- **Wizard de configuración** (4 pasos): tipo de modelo y backend → entrada
  (dimensiones, normalización, letterbox, layout/dtype) → salida (`pack_format`,
  estructura del tensor, NMS, anchors si corresponde) → escritura de
  `configs/<modelo>.json`.
- El **espejo (mirror)** se aplica solo a la cámara y solo en el cliente; los
  archivos subidos se muestran sin espejar.

### 3.2 Interfaz de programación (API HTTP)

Base: `http://127.0.0.1:8000`.

| Método | Ruta | Entrada | Salida / Comportamiento |
|---|---|---|---|
| GET | `/get_models` | — | Lista de configs que **tienen archivo de pesos** en `models/` (plantillas excluidas) |
| POST | `/select_model` | `{model_name}` | Carga atómica + validación post-carga. Errores honestos: 404 (no existe), 422 (config inválida), 501 (tipo no soportado), 500 (fallo real) |
| POST | `/model/load` | `{model_path}` | Igual que `select_model` pero por path directo |
| POST | `/model/unload` | — | Libera el pipeline |
| POST | `/config/confidence` | `{value: 0..1}` | Umbral en vivo. 409 sin modelo cargado, 422 fuera de rango |
| GET | `/logs/inference` | — | Últimos 50 errores de inferencia (buffer in-memory) |
| GET | `/metrics` | — | PerfMeter: avg/p95/FPS + desglose pre/inferencia/post (ventana de 300 muestras) |

Nota: `/config/colors` fue **eliminado**; los colores de dibujo viven en el cliente
(`overlay.js → drawSettings`).

### 3.3 Interfaz de streaming (WebSocket)

- Ruta: `ws://127.0.0.1:8000/video_stream`.
- **Entrada**: frames JPEG como mensajes binarios. Control de flujo en el cliente:
  **un frame en vuelo** a la vez.
- **Salida**: por **cada** frame recibido, un JSON:

```json
{ "detections": [[x1, y1, x2, y2, conf, cls], ...], "error": null }
```

- Errores posibles en el campo `error`: `"frame_invalido"`, `"no_model"`,
  `"inference_error"`. **El servidor responde siempre**, incluso ante fallo; el
  cliente mantiene además un timeout de 3 s como red de seguridad anti-deadlock.
- Las coordenadas llegan en **píxeles de la imagen original** enviada; el cliente
  repinta su propio frame y superpone las cajas.

### 3.4 Interfaz de configuración de modelos (JSON)

Cada modelo se describe en `configs/<nombre>.json`, validado por un esquema Pydantic
**estricto** (`extra="forbid"`): un campo desconocido o renombrado produce un error
de carga visible, nunca un silencio. Estructura raíz:

```
ModelConfig
├── model_type: "detection" | "classification" | "segmentation"   (solo detection operativo)
├── input:    dimensiones, normalize/mean/std/scale, letterbox + pad_color,
│             color_order (RGB/BGR/GRAY), input_str { layout HWC/CHW/NHWC/NCHW, dtype, quantized }
├── output (DetectionOutput):
│       pack_format: raw | yolo_flat | boxes_scores | tflite_detpost | anchor_deltas
│       tensor_structure: box_format (xyxy/cxcywh/yxyx), índices de columnas, num_classes
│       apply_conf_filter, confidence_threshold, apply_nms, nms_threshold, nms_per_class, top_k
│       anchor_config (REQUERIDO si pack_format == "anchor_deltas"):
│             min/max_level, num_scales, aspect_ratios, anchor_scale, box_variance,
│             scores_activation: none | sigmoid | softmax
└── runtime:  backend (onnxruntime/tflite/tensorflow/pytorch), device cpu/gpu,
              threads, warmup, opciones por runtime (providers ONNX, delegates TFLite),
              runtimeShapes.out_coords_space: normalized_0_1 | tensor_pixels
```

Reglas del contrato:

- `anchor_deltas` SIEMPRE entrega coordenadas en `tensor_pixels` (el JSON debe
  declararlo así).
- `tflite_detpost` ya trae NMS y umbral aplicados por el op de TFLite; el postproceso
  los desactiva por defecto para ese formato.
- `boxes_scores` entrega directamente el formato estándar; los demás formatos pasan
  por el `output_adapter` que reordena columnas.

### 3.5 Interfaz de software interna (pipeline de inferencia)

Contrato central por frame:

```
JPEG ─WS→ mainAPI → ModelController.inference(img_rgb)
  1. preprocess_fn → (tensor, meta)   letterbox/resize + normalización fusionada;
                                      meta = dict POR FRAME (orig_w/h, scale/pads del letterbox)
  2. input_adapter      color_order + layout + dtype
  3. predict_fn         inferencia del backend (devuelve numpy, nunca listas)
  4. unpack_fn          tensor crudo → matriz (N,6) float32
  5. output_adapter     reordena a [x1,y1,x2,y2,conf,cls] (solo formatos que lo requieren)
  6. postprocess_fn(arr, meta)  filtro de confianza → top-k → NMS → deshace letterbox → ordena por score
←WS─ JSON {detections, error}
```

El `meta` viaja junto al frame (no hay estado mutable compartido entre frames):
`runtime.runtimeShapes` contiene solo constantes de carga (`input_width/height`,
`out_coords_space`, tabla de anchors) escritas una única vez en `load_model`.

---

## 4. Requisitos funcionales

### 4.1 Gestión de modelos

- **RF-01** — El sistema debe listar como disponibles únicamente los modelos que
  tienen **tanto** un JSON de configuración en `configs/` **como** un archivo de
  pesos en `models/`. Las plantillas (`configs/plantillas/`) quedan excluidas.
- **RF-02** — El sistema debe cargar un modelo de forma **atómica**: el pipeline
  completo se construye antes de hacer commit al estado del controlador; ante
  cualquier fallo el controlador queda **descargado** y la excepción se propaga.
- **RF-03** — Tras la carga, el sistema debe ejecutar `validate_pipeline()`: una
  inferencia dummy end-to-end que detecta discrepancias entre el contrato declarado
  en el JSON y el modelo real, antes de aceptar frames.
- **RF-04** — El sistema debe rechazar la carga con códigos HTTP específicos:
  404 (modelo inexistente), 422 (configuración inválida), 501 (tipo de modelo no
  soportado: clasificación/segmentación), 500 (fallo interno), siempre con el detalle
  real del error.
- **RF-05** — El sistema debe permitir descargar el modelo activo, liberando el
  pipeline (`/model/unload`).
- **RF-06** — El frontend debe permitir importar archivos de pesos a `models/`
  mediante dropzone, aceptando `.onnx`, `.tflite`, `.h5`, `.keras`, `.pt`, `.pth`.
- **RF-07** — El sistema debe soportar los backends de inferencia `onnxruntime`,
  `tflite`, `tensorflow` (Keras) y `pytorch` (TorchScript), seleccionados por el
  campo `runtime.backend` del JSON.

### 4.2 Inferencia y streaming

- **RF-08** — El backend debe aceptar frames JPEG binarios por WebSocket y responder
  **un JSON por frame recibido, sin excepción**, incluso ante frame inválido
  (`error: "frame_invalido"`), sin modelo cargado (`error: "no_model"`) o fallo de
  inferencia (`error: "inference_error"`).
- **RF-09** — Toda detección devuelta debe estar en el **formato estándar**
  `[x1, y1, x2, y2, conf, cls]` en píxeles de la imagen original (con letterbox ya
  deshecho), ordenada por score.
- **RF-10** — El postprocesado debe aplicar, en orden: filtro de confianza → top-k →
  NMS (global o por clase, según config) → des-letterbox. Para `tflite_detpost`,
  filtro y NMS quedan desactivados por defecto (ya los aplica el op del modelo).
- **RF-11** — El sistema debe soportar modelos anchor-based con salida cruda
  (EfficientDet/SSD): generar la tabla de anchors al cargar
  (`anchor_gen.generate_efficientdet_anchors()`) a partir de `anchor_config`, aplicar
  los deltas con `box_variance` y la activación de scores declarada
  (`none`/`sigmoid`/`softmax`).
- **RF-12** — En cada inferencia, el sistema debe validar que los índices declarados
  en `tensor_structure` caben en el ancho real del tensor desempaquetado.
- **RF-13** — La inferencia debe ejecutarse en un threadpool para no bloquear el
  event loop del servidor; el acceso a load/inference/unload está serializado con un
  `RLock`.

### 4.3 Captura y visualización (frontend)

- **RF-14** — El frontend debe capturar de webcam (con selector de cámara) o de
  archivos de video/imagen, y enviar los frames por WS manteniendo **como máximo un
  frame en vuelo**, con timeout de 3 s por respuesta.
- **RF-15** — El espejado (mirror) debe aplicarse **solo en el cliente y solo para la
  cámara**; los archivos subidos se procesan y muestran sin espejar. El backend no
  voltea imágenes.
- **RF-16** — El frontend debe dibujar localmente el frame que envió más las cajas y
  etiquetas recibidas (`overlay.js`), con colores configurables en el cliente
  (`drawSettings`). Las etiquetas muestran actualmente el **id numérico de clase**
  (no hay `label_map` en detección — nota abierta).
- **RF-17** — El frontend debe permitir grabar el canvas de salida (video con las
  detecciones dibujadas) mediante MediaRecorder.

### 4.4 Configuración en vivo

- **RF-18** — El sistema debe permitir cambiar el umbral de confianza durante el
  streaming (`POST /config/confidence`); el postprocesador lee el valor **en cada
  llamada** (efecto inmediato). Debe responder 409 sin modelo cargado y 422 si el
  valor sale de [0,1].

### 4.5 Creación de configuraciones

- **RF-19** — El wizard del frontend debe generar `configs/<modelo>.json` en 4 pasos,
  cubriendo entrada, salida (incl. `anchor_config` cuando aplique) y runtime. La
  escritura pasa por IPC al proceso principal, que valida el nombre del archivo
  (anti path-traversal). *Limitación actual*: el contenido **no se valida contra el
  esquema del backend** al guardar; los defaults están duplicados respecto a
  `config_schema.py` (reforma pendiente #9).
- **RF-20** — El backend debe validar todo JSON de configuración contra el esquema
  estricto al cargar: campo desconocido = error 422 visible.

### 4.6 Observabilidad

- **RF-21** — El sistema debe exponer métricas de rendimiento (`GET /metrics`):
  promedio, p95 y FPS, con desglose pre/inferencia/post, sobre una ventana de 300
  muestras (PerfMeter).
- **RF-22** — El sistema debe exponer los últimos 50 errores de inferencia
  (`GET /logs/inference`, buffer in-memory).
- **RF-23** — El sistema debe mantener un log rotativo **por modelo** en
  `logs/<modelo>.log` (512 KB × 3 archivos, ~1.5 MB máx.); los logs por frame se
  emiten gateados cada 60 frames para no penalizar el hot path.

---

## 5. Requisitos no funcionales

### 5.1 Rendimiento

- **RNF-01** — El hot path de inferencia no debe introducir conversiones costosas:
  los `predict_fn` devuelven arrays numpy (nunca `.tolist()`); el preprocesado fusiona
  escala y normalización; los frames viajan binarios por WS (sin base64 ni doble
  compresión JPEG).
- **RNF-02** — El control de flujo (1 frame en vuelo) debe evitar acumulación de
  latencia: el cliente descarta frames en lugar de encolarlos.
- **RNF-03** — Soporte opcional de GPU: ONNX Runtime con CUDA 11.8
  (`onnxruntime-gpu==1.17.1`) y delegates configurables en TFLite.

### 5.2 Fiabilidad y robustez

- **RNF-04** — Invariante crítica: **el WS siempre responde** un mensaje por frame.
  Romperla reintroduce el deadlock del stream (bug histórico #3).
- **RNF-05** — Fallos de carga nunca dejan el controlador en estado intermedio
  (carga atómica, bug histórico #1).
- **RNF-06** — Los errores se reportan con su causa real (códigos HTTP específicos y
  `detail` informativo), nunca silenciados.

### 5.3 Mantenibilidad

- **RNF-07** — Convenciones de código: comentarios/docstrings
  en Python; estilo de builders/closures (`build_*`, `generate_*` devuelven
  funciones) para los pasos del pipeline; Prettier para JS (`npm run format`); no hay
  linter/formatter configurado para Python.
- **RNF-08** — Agregar un unpacker nuevo requiere tocar 3 lugares (registro en
  `unpackers/registry.py`, `Literal` del esquema, `<select>` del wizard) y decidir si
  entra en `_NEEDS_ADAPTER`.
- **RNF-09** — Suite de tests: **36 tests, todos verdes** (pytest con
  `pythonpath = src`), incluyendo un end-to-end con `models/yolov7-tiny.onnx`;
  `npx prettier --check "src/render/**/*.js"` para formato JS. No hay CI configurada
  (mejora pendiente #16).

### 5.4 Seguridad

- **RNF-10** — El backend escucha solo en `127.0.0.1` (no expuesto a la red).
- **RNF-11** — El renderer de Electron corre aislado: `contextIsolation: true`,
  `sandbox: true`, sin `nodeIntegration`. El único puente con el sistema es
  `window.uncaAPI` (4 operaciones expuestas por `preload.js` vía `contextBridge`);
  los handlers IPC del proceso principal validan los nombres de archivo
  (anti path-traversal) y revalidan las extensiones de modelos importados.

### 5.5 Portabilidad

- **RNF-12** — Desarrollado y operado sobre Windows; el backend es Python estándar y
  el frontend Electron, por lo que la portabilidad es plausible pero **no está
  verificada** en otros SO.

---

## 6. Limitaciones conocidas y deuda técnica (estado al 2026-06-12)

Resumen del registro vivo. Los 12 bugs de la ronda 2026-06-11
están **resueltos** (carga atómica, deadlock del WS, espejo de archivos, plantillas
listadas como modelos, EfficientDet con cajas basura, etc.).

### Notas abiertas (menores)

| # | Nota |
|---|---|
| A-1 | `top_k: int = False` en `config_schema.py` usa bool como default de int (los JSON ya usan `0`). |
| A-2 | `DetectionOutput` no tiene `label_map`: las etiquetas dibujadas muestran el id numérico de clase. |

### Reformas aplicadas el 2026-06-12

| # | Reforma |
|---|---|
| 6 | Hardening de Electron: `contextIsolation` + `sandbox` + `preload.js`/`contextBridge`; todo el `fs` del frontend movido a handlers IPC validados en el proceso principal. De paso se corrigió el dropzone (`File.path` no existe en Electron 32 → `webUtils.getPathForFile`). |
| 8 | Estado mutable por frame eliminado: `preprocess → (tensor, meta)` y `postprocess(arr, meta)`; `RuntimeShapes` quedó solo con constantes de carga. |

### Reformas pendientes (prioridad alta → baja)

| # | Reforma |
|---|---|
| 7 | Controller como administrador de pipelines multi-tipo (detección/clasificación/segmentación); unificar la normalización de shapes en una sola capa. |
| 8b | Achicar el `RLock` del controller (hoy serializa inferencias completas) para streams concurrentes reales, ahora que el estado compartido no existe. |
| 9 | Un solo origen de defaults para configs (endpoint de plantilla generado por Pydantic + validación server-side antes de escribir). |
| 11 | Pinear dependencias (`requirements.txt` sin versiones sobre Python 3.8 congelado). |
| 13 | Limpieza del repo (`.docx`, imágenes de prueba, binarios grandes de `models/` → git-lfs/.gitignore). |
| 14 | Arranque/apagado del backend desde Electron (spawn de uvicorn en `main.js`). |
| 15 | Snapshot de métricas por sesión en el log del modelo. |
| 16 | CI mínima (pytest unitarios + prettier --check). |

---

### Idea de ejecucion de sistema multi-tipo:

| Idea del model controller | En las ultimas reformas `model_controller` se convirtio en un operador con logica, enves de ser un orquestador abstraido de la logica con la que se opera en el sistema por dentro. Como idea para la restructuracion y administracion de pipelines multitipo propongo 


## 7. Apéndice — Entorno de ejecución

```bash
# Instalación
npm install
python -m venv .venv          # Python 3.8.10
.venv\Scripts\activate
pip install -r requirements.txt

# Terminal 1 — backend
uvicorn api.mainAPI:app --host 127.0.0.1 --port 8000 --app-dir src

# Terminal 2 — frontend
npm start

# Tests
pytest                                                          # 36 tests (incluye e2e)
pytest --ignore=src/api/func/tests/test_end_to_end_yolov7.py    # solo unitarios
npx prettier --check "src/render/**/*.js"
```

Dependencias principales: FastAPI + uvicorn + websockets; tensorflow,
onnxruntime-gpu 1.17.1, torch ≥ 2.0; Pillow, numpy, opencv-python. Frontend:
Electron ^32.3.1, Prettier ^3.4.2.
