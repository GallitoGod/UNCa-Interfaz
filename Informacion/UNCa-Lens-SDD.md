*UNCa-Lens \- 2026* 

# 

# Descripción de diseño de software (SDD) (Basado en IEEE 1016-2009)

---

Información general del documento

| Campo | Información |
| :---- | :---- |
| Proyecto del que forma parte | UNCa-Lens |
| Tipo de subsistema | Front-end |
| Versión del documento | 1.1 |
| Fecha | 26/06/2026 |
| Autor | Gallo Pablo Javier |

---

# 1.Introducción

El presente documento de descripción de diseño de software (SDD) tiene como propósito definir y documentar la arquitectura técnica, los componentes, el diseño de datos y las interfaces de usuario de la parte del Front-end del sistema UNCa-Lens.  
Mientras que el documento de especificaciones de requisitos de software (ERS) define qué debe hacer el sistema, este SDD detalla cómo se construirá, particularmente su parte Front-end. Su objetivo es servir como guía técnica principal para el desarrollo, traduciendo los requisitos funcionales y no funcionales específicos del área UX, UI de la ERS en una estructura, clara, modular y escalable, permitiendo la correcta implementación, pruebas y futuro mantenimiento del sistema bajo un enfoque de desarrollo Ágil e Incremental tomando al Back-end como una caja negra que obtiene, de, y devuelve, para, el Front-end datos e información necesaria para poder hacer funcionar el conjunto del sistema.

# 1.1 Propósito

Esta SDD sirve como un puente entre el documento ERS y la implementación. Su objetivo es: 

* Describir la arquitectura elegida (Thin Client, Feature-Driven Architecture) Detallar los componentes que conforman el sistema Front-end, indicando para cada uno sus responsabilidades, entradas, salidas, dependencias y reglas de negocio.  
* Documentar el diseño de entradas, salidas, interfaces y controles.

# 1.2 Alcance

El sistema Front-end cubre las siguientes funcionalidades:

* Electron: Actuar como contenedor nativo y puente con el Sistema Operativo, gestionando el ciclo de vida de la aplicación, el renderizado acelerado por hardware de la ventana principal y aplicando el endurecimiento de seguridad (hardening) mediante contextBridge e IPC restringido. EL ACCESO A DISCO NO ES PERMITIDO POR ELECTRON AL SER UN FRONT-END LIVIANO.  
* App-Shell: El “marco” de la aplicación. Arranque de renderer, navegación entre vistas, tabs de fuente, temas y la configuración del cliente Back-end. No cuenta con lógica de inferencia ni de modelos, solo orquesta y monta el resto.  
* Vision-Workspace: La superficie de presentación de resultados de inferencia en vivo. Es el componente del canvas, tiene una arquitectura de estrategias por tipo de modelo (detection / classification / segmentation) para que sea mantenible y expansible.  
* Wizard-inference: Esta sección cubre el transporte (WebSocket), las fuentes (cámara/archivo) y los controles.   
* Wizard-config: Módulo interactivo de recolección secuencial (form-wizard) responsable de capturar, estructurar y serializar los parámetros de inicialización de los modelos de visión. Carece de lógica de validación de Machine Learning; en su lugar, su capacidad crítica es la gestión de estados asíncronos de red, interceptando respuestas de error del Back-end (ej. HTTP 400/422) para decodificar el fallo, abortar el flujo de guardado y redirigir visualmente al usuario al campo de entrada específico que causó el conflicto.

---

# 2\. Visión general del diseño

El sistema Front-end de UNCa-Lens permitirá la puesta en producción de modelos de visión por computadora teniendo la capacidad de actualizarse en un futuro uniendo nuevas features como lo puede ser contadores de objetos X, velocidad de movimiento de un objeto X, etc. Desde un punto de vista técnico el objetivo de este sistema es actuar como un cliente liviano que solo recibe y devuelve datos sin entrar a memoria por fuera de las APIs del Back-end, lo que quiere decir que está terminantemente prohibido el uso de disco desde window en Electron.

# 2.1 Contexto del sistema

Este sistema no cuenta con un acceso de seguridad ni tiene previsto crear diferentes usuarios para su uso. Es un sistema técnico que no tiene un carácter comercial ni necesita ocultar datos dependiendo de qué usuario esté utilizando. Al mismo tiempo, no es un sistema simple y accesible para usuarios que no tienen conocimiento en los modelos utilizados, está pensado para ser utilizado por usuarios con las capacidades y conocimientos suficientes sobre el área de las redes neuronales convolucionales.

---

# 3\. Diseño arquitectónico

La arquitectura global del Front-end será un Cliente Liviano (Thin Client) bajo un diseño orientado a características (Feature-Driven Architecture).

1. Cliente liviano: Significa que el Front-end es “tonto”. Su única responsabilidad es pintar píxeles en la pantalla y capturar clicks/teclas. Cero lógica de procesamiento de vídeo bruto, cero validación de pesos de modelo; cero Back-end.

2. Feature-Driven: El código no se divide en tipos de archivo (una carpeta inmensa de componentes, otra de estados), sino por “módulos de negocio” (el Wizard por ejemplo). Si un módulo falla o se reemplaza, el resto del sistema ni se entera.

# 3.1 Frameworks/librerías utilizados

Stack técnico utilizado en el desarrollo del subsistema Front-end de UNCa-Lens:

1. El core: React, Vite, TypeScript  
   * React tiene una naturaleza descriptiva muy simple de operar, actualizar y mantener.  
   * Vite da un tiempo de compilación en caliente casi instantáneo.  
   * TypeScript, la primicia aquí es que el Back-end es una caja negra, por lo que se necesitan contratos fuertemente tipados en el Front-end, siendo TypeScript el encargado de hacer fallar la compilación antes de que siquiera se pruebe la app.  
       
2. Gestión de estado global: Zustand  
   * Es minúsculo, no envuelve la aplicación en Providers, y permite acceder al estado de forma selectiva. Es útil para guardar el estado progresivo de Wizard sin re-renderizar todo el árbol de componentes.  
       
3. Comunicación con la “caja negra” (API): TanStack Query, Axios  
   * TanStack Query se encarga del cache, los reintentos automáticos y expone banderas limpias como ‘isLoading’ o ‘isError’ (el manejo de errores sale casi gratis).  
       
4. Estilos: Tailwind CSS  
   * Simplifica la personalización utilizando clases utilitarias directamente en componentes.

# 3.2 Justificación de arquitectura y stack técnico

El planteo de esta arquitectura con este stack técnico está basado en estas tres preguntas:

1. ¿Qué arquitectura se va a usar?   
   La arquitectura es Thin Client en un diseño de Feature-Driven.  
     
2. ¿Estos frameworks/librerías están hechos para esa arquitectura?   
   * React, Vite y TypeScript son el motor de renderizado “tonto”. TypeScript asegura que el contrato de datos entre el Back-end (la caja negra) y el Front-end sea inquebrantable.  
   * TanStack Query es el portero de la arquitectura. En este cliente la comunicación con las APIs lo es todo. Esta librería aísla toda la lógica de peticiones, reintentos y mapeo de errores del Back-end para no ensuciar la UI.  
   * Zustand por la necesidad de estados locales complejos que no bloqueen el resto de la app. Entonces con Zustand se pueden crear mini-almacenes de memoria dedicados exclusivamente a cada feature.  
       
3. ¿Es capaz de operar al nivel de estrés que el sistema global requiere? 

	  
Para que este stack soporte el estrés del stream de video y las inferencias, el componente del Visión Workspace tiene que hacer un bypass al ciclo de vida de React. React solo se encargará de montar un elemento \<canvas\> vacío en el DOM. A partir de ahí, un script en JavaScript puro utilizará una referencia directa (useRef) para mutar el canvas usando la API nativa de HTML5 Canvas o WebGL, pintando los frames y los polígonos de forma totalmente desvinculada del árbol de React. 

---

# 4\. Diseño de componentes

1. Electron  
2. App-shell  
3. Wizard-inference  
4. Wizard-config  
5. Vision-workspace

# 4.1 Descripción de componentes

# 4.1.1 Electron

| Campo | Descripción |
| :---- | :---- |
| Responsabilidad | Es la capa nativa, encargada de dar portabilidad entre distintos sistemas operativos |
| Entradas | Ciclo de vida de Electron (app.whenReady, activate, window-all-closed) |
| Salidas | La ventana principal, carga de HTML. |
| Dependencias | Electron |
| Reglas de negocio | Tiene determinantemente prohibido el uso de disco. Toda comunicación IPC mediante contextBridge debe estar envuelta en Promises y devolver un objeto estandarizado: { success: boolean, data?: any, error?: string }.  El front-end debe hacer await  y manejar el estado de carga/error en React. |

* Estructura:

electron/  
├── main.ts  
├── preload.ts  
└── ipc-handlers.ts  
src/shared/electron/  
└── uncaAPI.d.ts

# 4.1.2 App-shell

| Campo | Descripción |
| :---- | :---- |
| Responsabilidad | Arranque del renderer, navegación entre vistas, configuración del cliente Back-end… Es un orquestador |
| Entradas | Evento DOMContentLoaded |
| Salidas | arranque del flujo de modelos y de cámara  |
| Dependencias | app/App.tsx, providers/, app/router.tsx, shared/api/axios.ts, ws.ts, shared/ui/Tabs, shared/ui/ThemeToggle |
| Reglas de negocio | el ModelSelector dispara la auto-selección del primer modelo al resolver la query (ver feature-inferencia.md), no en el bootstrap.  La feature Modelos se carga perezosamente (code-splitting). navegación por estado en un uiStore (Zustand) liviano, sin URLs (la app no las usa hoy). Al pasar de Inferencia a Modelos, el stream de video debe pasarse así como el WebSocket, reanudándose al volver. Ningún componente llama fetch/axios directo, se utiliza api.\* dentro de los hooks de Tanstack Query de cada feature. |

* Estructura:

src/  
├── app/  
│   ├── App.tsx                      \# monta providers \+ router  
│   ├── router.tsx                   \# ViewRouter (Inferencia | Modelos)  
│   └── providers/  
│       └── AppProviders.tsx   \# QueryClientProvider \+ (theme)  
└── shared/  
    ├── api/  
    │   ├── axios.ts                  \# instancia Axios (baseURL desde env)  
    │   └── ws.ts                      \# URL del WebSocket desde env  
    └── ui/  
        ├── Tabs.tsx  
        ├── ThemeToggle.tsx  
        └── Modal.tsx

# 4.1.3 Wizard-inference

| Campo | Descripción |
| :---- | :---- |
| Responsabilidad | Transporte por WebSocket, fuente como (cámara, archivo) y controles |
| Entradas | Mensajes WS entrantes: JSON { detections: \[\[x1,y1,x2,y2,conf,cls\], ...\] , error } (Esa es la entrada actual del WebSocket antes de ser capaz de utilizar modelos de clasificación y segmentación) Frames de canvas de captura. |
| Salidas | Frames JPEG binarios al backend. |
| Dependencias | (1)services/videoStream, (2)components/CameraSource.tsx, (3)components/FileSource.tsx,  (4)components/ModelSelector.tsx \+ useModels |
| Reglas de negocio | (1)Transporte WS framework-agnóstico (no es un hook; no re-renderiza). Captura frames, los envía como JPEG binario y emite los payloads crudos. El consumidor (VisionWorkspace) hace parse/present. 1 frame en vuelo: no se envía el siguiente hasta recibir respuesta (waitingForResponse). (1)Anti-deadlock: si no llega respuesta en RESPONSE\_TIMEOUT\_MS \= 3000, libera la espera. (1)El frame enviado queda intacto en el captureCanvas hasta la próxima captura: por eso onMessage recibe ese canvas para que el workspace repinte exactamente el frame que produjo el resultado. (1)Reconexión con backoff exponencial (1s → máx 10s); no reconecta si el cierre fue intencional (close()). (1)mirror se aplica en la captura (scale(-1,1)), solo cuando lo pide la cámara. (1)Envío binario (ws.send(blob), JPEG calidad 0.8); nada de base64. (1)No usar TanStack Query (es un stream long-lived, no request/response).  (2)Espejo solo cámara. Al cambiar de cámara, cerrar el stream anterior y detener los tracks previos antes de abrir el nuevo (evita fugas). (2)Pedir permiso con getUserMedia antes de poblar la lista (si no, los label vienen vacíos); tras refrescar, intentar mantener la cámara previa. (2)Exponer un stopCurrentStream() (en el hook/store) para que FileSource libere la cámara. (2)La lista de dispositivos es un hook propio (useCameras), no TanStack Query (es API del browser). Device activo \+ mirror en streamStore. (3)Video: video.src \= URL.createObjectURL(file), loop+muted, reusa startVideoStream con mirror:false. (3)Imagen: WS one-shot dedicado (un frame calidad 0.9, dibuja, cierra) — no pasa por el loop. Se factoriza como sendSingleFrame(blob) en videoStream.ts. (3)Antes de iniciar: stopCurrentStream() (libera la cámara) y cerrar cualquier WS de archivo previo. Sin espejo. (4)Al resolver la query, auto-seleccionar el primer modelo (efecto en el componente). (4)/get\_models solo lista configs con pesos (filtro del backend). (4)Importante (cross-feature): al seleccionar un modelo, además de POST /select\_model, el selector debe leer el model\_type del config del modelo y setearlo en el workspaceStore (setActiveModel(name, type)), porque el vision-workspace enruta por ese tipo. El tipo sale del Back-end.  |

* Estructura:

features/inference/  
├── InferenceView.tsx             \# compone fuentes \+ VisionWorkspace \+ controles  
├── services/  
│   └── videoStream.ts            \# transporte WS (framework-agnóstico)  
├── store/  
│   └── streamStore.ts             \# estado del stream (conectado, esperando, último error)  
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
│   └── useModels.ts              \# get\_models \+ select\_model  
└── api/  
    ├── models.ts                     \# GET /get\_models, POST /select\_model  
    └── confidence.ts               \# POST /config/confidence

* Controles Hooks de Query/mutation

| Componente | Hook | Implementación |
| :---- | :---- | :---- |
| ConfidenceSlider | useUpdateConfidence | useMutation → POST /config/confidence. Slider \[0,100\] → /100. Debounce en lugar del change-only actual; valor local en el componente. |
| MetricsHUD | useMetrics | useQuery con refetchInterval: 1000, enabled atado a si el HUD está abierto. Reemplaza el setInterval. \-- si status \!== 'ok'. |
| LogPanel | useInferenceLogs | useQuery con refetchInterval: 5000, enabled atado a abierto. Render en JSX (elimina el innerHTML actual). Orden inverso (más reciente primero). |
| Recorder | useRecorder | canvas.captureStream(30) \+ MediaRecorder. Regla estricta: Interceptar el Blob resultante al finalizar la grabación y utilizar la librería fix-webm-duration para inyectar la metadata de duración antes de generar el archivo final. |
| DrawSettingsModal | — | shared/ui/Modal; escribe en drawSettingsStore. Commit de colores solo al Guardar. Mejora: persistir en localStorage. |

# 4.1.4 Wizard-config

| Campo | Descripción |
| :---- | :---- |
| Responsabilidad | Muestra las tarjetas de pesos disponibles, dropzone para importar nuevos, y el wizard de configuración de 4 pasos que arma el JSON del config de cada modelo. Tiene que listar los pesos de models/ como tarjetas (con estado "tiene config / sin config") y, al hacer click en una, abrir el wizard con la config existente o con defaults.  |
| Entradas | Lista de modelos desde el Back-en de la forma { ok, models: \[{ file, ext, baseName, hasConfig }\] }. Configuración del modelo desde el Back-end de la forma { ok, config } (null si no existe la configuración). Click en tarjeta, click en refrescar. |
| Salidas | Renderiza el grid de tarjetas, marca la tarjeta seleccionada |
| Dependencias | (1)ModelsCrig \+ ModelCard, (2)useImportModels, (3)ConfigWizard |
| Reglas de negocio | (1)No toca el disco directamente: todo pasa por Back-end. (1)Si readConfig falla (JSON corrupto), abre el wizard con defaults igual (no bloquea). (1)Estados vacíos/errores explícitos ("No hay modelos…", "No se pudo leer models/"). (1)Tras guardar en el wizard → queryClient.invalidateQueries(\['models'\]) (reemplaza el re-escaneo manual). (2)extensiones aceptadas .onnx/.tflite/.h5/.keras/.pt/.pth. (3)El wizard pide los defaults al backend, que ya es la single source of truth. (3)Al abrir el wizard para un modelo sin config, se usa la plantilla del backend como config inicial. Para uno con config existente, se carga la suya. (3)Un componente por paso: Step1Type, Step2Input, Step3Output/{Detection,Classification, Segmentation}Output, Step4Runtime. El paso 3 se elige por config.model\_type. (3)Los helpers num/numf/sel/chk/txt se vuelven componentes controlados (NumberField, FloatField, SelectField, CheckField, TextField) que leen del store y escriben con setField(path, value). (3)Campos condicionales \= render declarativo (sin toggleHidden/innerHTML): se muestran con {cond && \<Section/\>}. Casos: normalize→mean/std; letterbox→pad color (solo detection); apply\_nms→nms\_threshold; pack\_format \=== 'anchor\_deltas'→sección anchors; backend→sección ONNX/TFLite; box\_format→llaves de coordenadas. (3)Letterbox y anchors solo para detection. (3)Guardar por POST /configs/{name} (mutation useSaveConfig), que valida contra ModelConfig estricto en el backend antes de escribir. (3)ModelConfig y sus sub-tipos (input, output, runtime, anchor\_config, …) se derivan del schema Pydantic estricto (config\_schema.py) y viven en shared/api/types.ts. Así el wizard no puede armar un config con un campo que el backend va a rechazar.  |

* Estructura:

features/models/  
├── ModelsView.tsx  
├── components/  
│   ├── ModelsGrid.tsx  
│   ├── ModelCard.tsx  
│   ├── ModelDropzone.tsx  
│   └── ConfigWizard/  
│       ├── ConfigWizard.tsx       \# stepper \+ footer \+ navegación  
│       ├── Step1Type.tsx  
│       ├── Step2Input.tsx  
│       ├── Step3Output/  
│       │   ├── DetectionOutput.tsx  
│       │   ├── ClassificationOutput.tsx  
│       │   └── SegmentationOutput.tsx  
│       ├── Step4Runtime.tsx  
│       └── fields/                        \# NumberField, FloatField, SelectField, CheckField, TextField  
├── store/  
│   └── wizardStore.ts               \# estado del wizard (4 pasos)  
├── hooks/  
│   ├── useModels.ts                 \# uncaAPI.listModels (IPC)  
│   ├── useModelConfig.ts        \# uncaAPI.readConfig (IPC)  
│   ├── useImportModels.ts      \# uncaAPI.importModels (IPC)  
│   ├── useConfigTemplate.ts   \# GET /config/template/{model\_type}  
│   └── useSaveConfig.ts         \# POST /configs/{name}  
├── api/  
│   └── configs.ts                      \# template \+ save (Axios)  
└── lib/  
    └── toBackendConfig.ts       \# transformación pura previa al guardado  
(testeable)

* Testing

| Qué | Cómo |
| :---- | :---- |
| toBackendConfig | Función pura: casos detection/classification/segmentation \+ cada transformación (mover out\_coords\_space, anular backends, anular anchor\_config). El de mayor valor. |
| setField/setModelType | Store: asignación por path y reseteo de output al cambiar tipo. |
| Campos condicionales | Render: que las secciones aparezcan/desaparezcan según el campo gatillo. |
| useConfigTemplate | Que el wizard arranque con los defaults del backend, no hardcodeados. |

# 4.1.5 Vision-Workspace

| Campo | Descripción |
| :---- | :---- |
| Responsabilidad | montar el \<video\>, el canvas de salida y un contenedor de overlays HTML; por cada mensaje del stream, repintar el frame base y delegar la presentación del resultado en la estrategia del tipo de modelo activo.  |
| Entradas |  |
| Salidas | Video canvas |
| Dependencias | (1)services/registry.ts, (2)store/workspaceStore.ts, (3)detection.service.ts, (4)classification.service.ts, (5)segmentation.service.ts |
| Reglas de negocio | El workspace repinta el frame base; la estrategia solo agrega su capa encima (nunca repinta el frame). Así, detección (cajas), segmentación (máscara) y clasificación (badge HTML) componen sin pisarse y se pueden combinar a futuro.  (1)Agregar un tipo nuevo en el futuro \= crear \<tipo\>.service.ts y registrarlo acá. (2)El model\_type se setea al seleccionar modelo (lo provee la feature Modelos / el ModelSelector leyendo el config). drawSettings lo edita el DrawSettingsModal.  (3)Coordenadas en px de la imagen original (el backend ya deshizo el letterbox). (3)La etiqueta muestra el id numérico de clase salvo que llegue labelMap (futuro). (3)Si la caja toca el borde superior, la etiqueta se dibuja dentro de la caja. (4)Actualmente no se encuentra diseñado, tiene un checkList pendiente (ver abajo). (5)Actualmente no se encuentra diseñado, tiene un checkList pendiente (ver abajo). |

* Estructura:

features/vision-workspace/  
├── components/  
│   ├── VisionWorkspace.tsx          \# monta video+canvas+overlayRoot; orquesta el loop  
│   └── UnsupportedOverlay.tsx     \# "este tipo aún no tiene visualización"  
├── services/  
│   ├── types.ts                               \# VisionStrategy, VisionFrameContext, DrawSettings, ModelType  
│   ├── registry.ts                            \# mapa model\_type \-\> strategy  
│   ├── detection.service.ts            \# IMPLEMENTADO (migra overlay.js)  
│   ├── classification.service.ts      \# STUB (implemented:false)  
│   └── segmentation.service.ts     \# STUB (implemented:false)  
├── store/  
│   └── workspaceStore.ts             \# modelo activo {name, type}, drawSettings, fuente activa  
└── hooks/  
    └── useVisionWorkspace.ts        \# conecta el stream y corre parse/present con la estrategia activa

* CheckList classification.service.ts:  
    
1. Definir el payload real del WS para clasificación (acordar con el backend).  
2. parse: mapear payload → ClassificationResult (aplicar top-k client-side si hace falta, o confiar en el backend).  
3. present: crear/actualizar un nodo HTML en overlayRoot (badge con clase \+ score). Considerar labelMap para mostrar nombres en vez de ids.  
4. clear: remover el badge.  
5. Flip implemented: true.

* CheckList segmentation.service.ts:  
    
1. Definir cómo viaja la máscara por el WS (forma \+ codificación; una máscara por píxel es pesada → evaluar RLE/PNG/base64 vs ndarray plano).  
2. parse: payload → SegmentationResult.  
3. present: construir un ImageData/canvas offscreen coloreando por settings.colormap, y drawImage sobre el frame con settings.maskAlpha.  
4. Considerar output\_stride/resize\_to\_input (redimensionar la máscara al frame).  
5. Flip implemented: true.

* Manejo de errores:

  * Tipo no soportado (implemented:false): el workspace muestra UnsupportedOverlay ("Clasificación: visualización aún no implementada") y no intenta parsear/presentar. Sin throw en el hot path. Consistente con el 501 honesto del backend.  
      
  * payload.error: "no\_model" → frame solo (normal); otro valor → log a consola (los detalles están en /logs/inference).  
      
  * Fallo en present: envuelto en try/catch por frame; un error puntual no rompe el loop ni la conexión.

### 

* Testing


| Qué | Cómo |
| :---- | :---- |
| detection.parse | Función pura: payloads de ejemplo → Detection\[\] |
| detection.present | Canvas mock (jsdom/OffscreenCanvas): verificar que se llaman strokeRect/fillText. Opcional. |
| Stubs CLS/SEG | Afirmar implemented \=== false, parse() \=== null, y que el workspace muestra UnsupportedOverlay al seleccionarlos. |
| Despacho | getStrategy(type) devuelve la estrategia correcta por tipo. |

---

# Estado de implementación (2026-06-26)

Esta sección documenta lo **realmente construido** y resuelve las dos contradicciones
internas que tenía la v1.0 del SDD. El frontend nuevo vive en `client/src/` (Feature-Driven:
`app/`, `features/`, `shared/`), sobre la rama `refactor-frontend-react`. Los 5 componentes
están implementados; verificación: backend **52/52** tests verde, `typecheck` + `vite build`
verdes. Pendiente transversal: **verificación visual en Electron** (cámara real, grabación,
pausa/reanudación, upload, banner de error del wizard) — no automatizable.

## Contradicciones de la v1.0 resueltas

1. **Acceso a disco (§2 vs §4.1.4).** El §2 prohíbe el disco vía Electron, pero el §4.1.4
   y el Anexo A.1 todavía describían IPC (`listModels`/`readConfig`/`importModels`/
   `writeConfig`). **Se resolvió a favor de SIN-DISCO:** el IPC de disco se **eliminó** por
   completo (se borró `uncaApi.ts`; `ipc-handlers.js` y `preload.js` quedaron no-op). Toda la
   persistencia va por HTTP al backend:
   - `GET /models` — lista de pesos con `hasConfig` (vista Modelos).
   - `GET /configs/{name}` — leer config existente (faltante→`{config:null}`, corrupto→500).
   - `POST /configs/{name}` — validar (ModelConfig estricto) y escribir.
   - `POST /models/upload` — subir pesos por multipart (valida extensión + nombre seguro,
     escribe en chunks). El dropzone sube los `File` del browser (sin `getPathForFile`).

   El Anexo A.1 (IpcResponse / `window.uncaAPI`) queda como **convención para IPC futuro
   NO-disco** (hoy no hay ningún canal IPC vivo). Efecto colateral: la vista Modelos funciona
   también en un browser de dev (sin Electron).

2. **Contrato del WebSocket (Anexo A.3 vs §4.1.3/§4.1.5).** El Anexo A.3 propone un sobre
   nuevo (`{ error, type, timestamp, results }`, coords normalizadas 0–1) pero el §4.1.3/§4.1.5
   piden "sin tocar el WS" y el backend ya entrega **píxeles** (deshace el letterbox). **Se
   mantiene el contrato actual** `{ detections: [[x1,y1,x2,y2,conf,cls]], error }` en píxeles;
   el Anexo A.3 queda como **visión futura** (se autodeclara aproximado). El enrutado por
   `model_type` se hace client-side sin envolver el WS.

## Estado por componente

- **Electron (§4.1.1):** contenedor de ventana con hardening intacto; `main.js` carga el
  **build de Vite** (`client/dist/index.html` en prod; dev server con `electron . --dev`). Sin
  IPC de disco. Los 3 archivos siguen en `src/` como **JS (CommonJS)**, no migraron a TS.
- **App-shell (§4.1.2):** `app/` (App, AppProviders=QueryClient, router lazy, `uiStore`
  vista+tema **persistido**). **Pausa/reanudación implementada:** `InferenceView` queda montada
  y oculta al ir a Modelos; la sesión se pausa (loop + `<video>` + `track.enabled=false`) sin
  cerrar el WS ni soltar permisos, y reanuda al volver (`videoStream.pause/resume` +
  `useVisionSession`).
- **Wizard-inference (§4.1.3):** transporte con las invariantes + `pause/resume`; fuentes
  cámara/archivo; `ModelSelector` lee el `model_type` **real** (`GET /configs/{name}`) y setea
  el workspace. Controles: slider con **debounce**, HUD/Logs con `refetchInterval`+`enabled`,
  **Recorder con `fix-webm-duration`** (hook `useRecorder`), DrawSettings **persistido en
  localStorage**. Hooks de diagnóstico consolidados en `useDiagnostics.ts`.
- **Wizard-config (§4.1.4):** wizard de 4 pasos, Step3 por tipo, campos controlados, todos los
  condicionales, defaults por `GET /config/template`, guardado por `POST /configs`. **Capacidad
  crítica (§1.2) implementada:** ante un 422 el wizard **decodifica el error, aborta el guardado,
  navega al paso del campo y muestra un banner que lo nombra** (clickable). Se decidió banner +
  navegación; el resaltado inline por campo quedó fuera (bajo retorno).
- **Vision-workspace (§4.1.5):** `present.ts` (regla de oro + errores del hot path),
  `registry` (despacho por tipo), `detection.service` implementado, CLS/SEG como stubs
  `implemented:false` con checklists, `UnsupportedOverlay`. El orquestador del loop es
  `useVisionSession` (no un `useVisionWorkspace` propio).

## Divergencias estructurales menores (vs los diagramas del SDD)

- Electron quedó en `src/*.js` (CJS), no en `electron/*.ts`.
- Los hooks de inferencia/modelos se consolidaron por feature (`useDiagnostics.ts`,
  `useModelsList.ts`) en vez de un archivo por hook.
- El código React está bajo `client/src/` (no `src/`), para no contaminar el `src/`
  compartido con el backend Python + Electron.

---

# Anexo A: Diccionario de datos compartidos (Interfaces TypeScript)

Este anexo define de forma estricta los contratos de datos (JSON) que fluyen entre la UI (React), la capa nativa (Electron IPC) y el Back-end (API Rest / WebSocket). Se ubicarán en src/shared/api/types.ts. (Este anexo puede tener fallos con relación al estado actual de los datos compartidos, es solo una idea para entender por dónde encaminarse)

### 

# A.1 Interfaces IPC (Comunicación Electron \- React)

> NOTA (v1.1): el IPC de disco fue **eliminado** (ver "Estado de implementación"). Estas
> interfaces quedan como **convención para IPC futuro NO-disco**; hoy no hay canales vivos.
> La persistencia de modelos/configs va por HTTP, no por `window.uncaAPI`.

Toda comunicación a través del contextBridge DEBE retornar una Promesa que resuelva en un objeto IpcResponse.

\`\`\`  
export interface IpcResponse\<T \= any\> {  
  success: boolean;  
  data?: T;  
  error?: string;  
}

// Ejemplo de uso para listModels: Promise\<IpcResponse\<ModelFile\[\]\>\>  
export interface ModelFile {  
  file: string;      // Ruta absoluta devuelta por el OS  
  ext: string;       // Extensión del archivo (.onnx, .pt, etc.)  
  baseName: string;  // Nombre sin extensión  
  hasConfig: boolean;// Flag que indica si existe un JSON asociado  
}  
\`\`\`

### 

# A.2 Interfaces de Configuración de Modelos (Wizard-config)

Estas interfaces reflejan la estructura exacta del JSON que el Back-end espera para inicializar un modelo.

\`\`\`  
export type ModelType \= 'detection' | 'classification' | 'segmentation';

// Estructura base requerida por el backend (equivalente al Pydantic schema)  
export interface ModelConfig {  
  model\_type: ModelType;  
  input: InputConfig;  
  output: OutputConfig;  
  runtime: RuntimeConfig;  
}

export interface InputConfig {  
  width: number;  
  height: number;  
  normalize\_mean?: \[number, number, number\];  
  normalize\_std?: \[number, number, number\];  
}

export interface OutputConfig {  
  // Específico de Detección (DetectionOutput)  
  box\_format?: 'xyxy' | 'xywh' | 'cxcywh';  
  apply\_nms?: boolean;  
  nms\_threshold?: number;  
  confidence\_threshold: number;  
  // Específico de Clasificación (ClassificationOutput)  
  top\_k?: number;  
  // Específico de Segmentación (SegmentationOutput)  
  mask\_threshold?: number;  
}

export interface RuntimeConfig {  
  backend: 'onnx' | 'tflite' | 'pytorch';  
  device: 'cpu' | 'cuda' | 'xla';  
}  
\`\`\`

### 

# A.3  Interfaces de Inferencia en Vivo (WebSocket)

> NOTA (v1.1): el contrato **implementado** es el actual `{ detections: [[x1,y1,x2,y2,conf,cls]],
> error }` en **píxeles** de la imagen original. El sobre de abajo (con `type`/`timestamp`/
> `results` y coords normalizadas 0–1) es una **propuesta a futuro**, no lo que corre hoy
> (ver "Estado de implementación", contradicción #2).

Este es el contrato para los mensajes que transitan a altísima velocidad por el WebSocket durante el renderizado del Vision-Workspace.

\`\`\`  
// Estructura general del mensaje que llega por WebSocket  
export interface WSInferenceMessage {  
  error?: string;  
  type: ModelType;  
  timestamp: number;  
  results: InferenceResult\[\];   
}

// Tipo unión para manejar los resultados dependiendo del modelo activo  
export type InferenceResult \= DetectionResult | ClassificationResult | SegmentationResult;

export interface DetectionResult {  
  box: \[number, number, number, number\]; // Coordenadas normalizadas \[x\_min, y\_min, x\_max, y\_max\] (0 a 1\)  
  confidence: number;  
  class\_id: number;  
  label?: string; // Opcional, si el backend envía el nombre resuelto  
}

export interface ClassificationResult {  
  class\_id: number;  
  confidence: number;  
  label?: string;  
}

export interface SegmentationResult {  
  // Asumiendo envío optimizado como array 1D de bytes (Run-Length Encoding o similar)  
  mask\_data: number\[\];   
  width: number;  
  height: number;  
  class\_id: number;  
  confidence: number;  
}  
\`\`\`  
