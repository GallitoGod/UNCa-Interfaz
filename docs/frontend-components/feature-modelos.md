# Feature: Modelos

La vista "Modelos": tarjetas de los pesos disponibles, dropzone para importar nuevos, y
el **wizard de configuración de 4 pasos** (el "wizard-config") que arma el JSON de config
de cada modelo.

Archivos as-is: `modelsManager.js`, `configBuilder.js`.

> **Estado (actualizado 2026-06-26 — migración React):** las secciones *as-is* de abajo
> describen el frontend viejo (`modelsManager.js`, `configBuilder.js`), hoy **muerto**. El
> destino React ya está implementado en `client/` y, además, **el SDD prohíbe el acceso a
> disco desde Electron** (thin client). Por eso **toda la divergencia documentada acá quedó
> cerrada y SUPERADA**: el wizard ya no usa IPC ni defaults hardcodeados, y la feature
> entera dejó de tocar disco. Toda la persistencia va por HTTP:
> `GET /models`, `GET /configs/{name}`, `POST /configs/{name}`, `POST /models/upload`.
> El bloque "Diseño to-be" de abajo refleja lo realmente construido.

---

## ModelsManager

- **Responsabilidad:** Listar los pesos de `models/` como tarjetas (con estado "tiene
  config / sin config") y, al hacer click en una, abrir el wizard con la config existente
  o con defaults.
- **Entradas:**
  - `window.uncaAPI.listModels()` → `{ ok, models: [{ file, ext, baseName, hasConfig }] }`.
  - `window.uncaAPI.readConfig(baseName)` → `{ ok, config }` (config `null` si no existe).
  - Click en una tarjeta; click en "refrescar".
- **Salidas:** renderiza el grid de tarjetas; llama `openBuilder(...)`; marca la tarjeta
  seleccionada.
- **Dependencias:** `configBuilder.openBuilder`, `window.uncaAPI` (IPC),
  `#model-cards-grid`, `#config-builder`.
- **Reglas de negocio:**
  - **No toca el disco directamente:** todo pasa por `uncaAPI` (main process). Por eso las
    operaciones son async.
  - Si `readConfig` falla (JSON corrupto), abre el wizard con defaults igual (no bloquea).
  - Estados vacíos/errores explícitos ("No hay modelos…", "No se pudo leer models/").
  - Tras guardar en el wizard, **re-escanea** para refrescar el estado de las tarjetas.
- **Mapeo al destino React:** `features/models/components/ModelsGrid` + `ModelCard`; hook
  `useModels` (TanStack Query sobre `uncaAPI.listModels`) y `useModelConfig(baseName)`
  (query sobre `uncaAPI.readConfig`). El re-escaneo post-guardado se vuelve
  `queryClient.invalidateQueries`.

## ModelDropzone

- **Responsabilidad:** Importar modelos arrastrándolos: filtra por extensión y los copia a
  `models/` vía IPC.
- **Entradas:** eventos `dragover`/`dragleave`/`drop`; los `File` soltados.
- **Salidas:** `window.uncaAPI.getPathForFile(file)` (resuelve el path real) →
  `window.uncaAPI.importModels(paths)`; feedback visual; re-escaneo del grid.
- **Dependencias:** `window.uncaAPI` (IPC), `SUPPORTED_EXTENSIONS`.
- **Reglas de negocio:**
  - Extensiones aceptadas: `.onnx`, `.tflite`, `.h5`, `.keras`, `.pt`, `.pth`.
  - El filtro del renderer es **solo UX**: la validación real la repite el main process
    (`models:import`).
  - **`File.path` ya no existe** en Electron ≥ 32: el path real se resuelve en el preload
    con `webUtils.getPathForFile`.
  - Reporta cuántos se agregaron y cuántos se ignoraron.
- **Mapeo al destino React:** `features/models/components/ModelDropzone` +
  `useImportModels` (mutation TanStack Query envolviendo `uncaAPI.importModels`).
  `getPathForFile` sigue siendo del preload (Electron). Tailwind para los estados
  `drag-over`/feedback.

## ConfigWizard ("wizard-config")

El componente más grande del frontend (`configBuilder.js`, ~740 líneas). Wizard de 4
pasos que construye el JSON de configuración de un modelo.

- **Responsabilidad:** Guiar la creación/edición del config de un modelo en 4 pasos —
  **Tipo → Input → Output → Runtime** — y guardarlo. El contenido de cada paso depende del
  `model_type` (detection / classification / segmentation).
- **Entradas:**
  - `openBuilder(builderEl, modelFile, existing, onSave)`: nodo contenedor, nombre de
    archivo del modelo, config existente (o `null` → defaults) y callback de guardado.
  - Interacción del usuario en cada paso (inputs `number`/`text`/`select`/`checkbox`,
    tarjetas de tipo).
- **Salidas:**
  - `window.uncaAPI.writeConfig(baseName, config)` → escribe `configs/<baseName>.json`.
  - Feedback de éxito/error; invoca `onSave` (que dispara el re-escaneo del manager).
- **Dependencias:** `window.uncaAPI.writeConfig` (IPC), el DOM del contenedor
  `#config-builder`. Los defaults son **internos** (no consulta al backend).
- **Reglas de negocio:**
  - **Estado interno de módulo** (`_state`): `{ modelFile, step, config }`. Re-renderiza
    todo el HTML del builder en cada cambio de paso.
  - **Defaults por tipo** (`OUTPUT_DEFAULTS`) y comunes (`INPUT_DEFAULTS`,
    `RUNTIME_DEFAULTS`, `ANCHOR_DEFAULTS`) viven en el módulo — **duplican** el schema
    Pydantic del backend (deuda conocida #9 en CLAUDE.md).
  - **Campos condicionales:** secciones que aparecen/desaparecen según otros campos
    (normalize → mean/std; letterbox → pad color; apply_nms → nms_threshold;
    `pack_format == anchor_deltas` → sección de anchors; backend → sección ONNX/TFLite;
    `box_format` → re-genera las llaves de coordenadas).
  - **Letterbox y anchors solo para `detection`**; cada tipo tiene su propio paso 3.
  - **Transformaciones al guardar (`save`)**, alineadas al schema estricto del backend:
    - `output.out_coords_space` se mueve a `runtime.runtimeShapes.out_coords_space` (y se
      borra de `output`).
    - Se anula el backend no usado (`tflite`/`onnx` → `null` según `backend`).
    - `anchor_config` se anula salvo `pack_format == anchor_deltas`.
  - El nombre del archivo sale del `modelFile` quitando la extensión (`baseName`); la
    validación anti path-traversal la hace el main process.
- **Mapeo al destino React:** feature propia `features/models/components/ConfigWizard/`
  con un componente por paso (`Step1Type`, `Step2Input`, `Step3Output/*`, `Step4Runtime`)
  y los campos como componentes controlados (`NumberField`, `SelectField`, etc.,
  reemplazando los helpers `num/numf/sel/chk/txt`). El `_state` global pasa a un **store
  Zustand** (`wizardStore`) o `useReducer`. **Cambios clave de la migración:**
  - Consumir `GET /config/template/{model_type}` (hook `useConfigTemplate`) en vez de los
    `*_DEFAULTS` hardcodeados → cierra la divergencia documentada arriba.
  - Guardar por `POST /configs/{name}` (mutation) — **o** mantener `uncaAPI.writeConfig`
    si se decide que el guardado siga siendo local por IPC. **Decisión a tomar.**
  - La lógica de campos condicionales se vuelve renderizado declarativo (sin
    `toggleHidden`/`innerHTML`). Las transformaciones de `save` pasan a una función pura
    `toBackendConfig(state)` testeable.
  - Tipar el config contra el schema Pydantic (tipos en `shared/api/types.ts`).

---

## Acoplamiento con la plataforma (Electron)

El frontend viejo dependía de `window.uncaAPI` (IPC) para tocar disco. **En el destino React
ese acoplamiento se ELIMINÓ**: por mandato del SDD (thin client sin disco), la feature ya no
usa IPC para nada — listar/importar modelos y leer/escribir configs van por HTTP al backend.
El `uncaApi.ts` se borró y los handlers de disco del main process quedaron como no-op
(ver [`plataforma-electron.md`](./plataforma-electron.md)).

---

## Diseño to-be (React)

Profundidad escalada: medio para ModelsManager/Dropzone; **profundo para el ConfigWizard**
(es el componente más grande del frontend y el que más se beneficia del rediseño).

### Estructura

```
features/models/
├── ModelsView.tsx
├── components/
│   ├── ModelsGrid.tsx
│   ├── ModelCard.tsx
│   ├── ModelDropzone.tsx
│   └── ConfigWizard/
│       ├── ConfigWizard.tsx       # stepper + footer + navegación
│       ├── Step1Type.tsx
│       ├── Step2Input.tsx
│       ├── Step3Output/
│       │   ├── DetectionOutput.tsx
│       │   ├── ClassificationOutput.tsx
│       │   └── SegmentationOutput.tsx
│       ├── Step4Runtime.tsx
│       └── fields/                # NumberField, FloatField, SelectField, CheckField, TextField
├── store/
│   └── wizardStore.ts            # estado del wizard (4 pasos)
├── hooks/
│   └── useModelsList.ts          # useModelsList/useModelConfig/useImportModels (REST) +
│                                 #   useConfigTemplate/useSaveConfig
├── api/
│   ├── models.ts                # GET /models (listar), POST /models/upload (subir) + ModelEntry
│   └── configs.ts               # GET/POST /configs/{name}, GET /config/template (Axios)
└── lib/
    ├── toBackendConfig.ts        # transformación pura previa al guardado (testeable)
    └── validationErrors.ts       # decodifica el 422 del backend -> issues por-campo + paso
```

> **Nota:** los hooks quedaron consolidados en un solo `useModelsList.ts` (en vez de un
> archivo por hook) y **todos son REST** (TanStack Query + Axios). NO hay IPC: el SDD
> prohíbe el disco vía Electron, así que `listModels`/`readConfig`/`importModels` pasaron a
> `GET /models` / `GET /configs/{name}` / `POST /models/upload`. Efecto colateral: la vista
> Modelos ahora funciona también en un browser de dev (sin Electron).

### ModelsManager → `ModelsGrid` + `ModelCard` *(medio)*

```ts
// hooks/useModelsList.ts (REST, sin IPC)
export function useModelsList() {
  return useQuery({ queryKey: ['models-list'], queryFn: listModels }); // GET /models
}
export function useModelConfig(baseName: string | null) {
  return useQuery({
    queryKey: ['model-config', baseName],
    queryFn: () => getConfig(baseName!),   // GET /configs/{name}
    enabled: !!baseName,
  });
}
```

Reglas heredadas:

- **React nunca toca disco**: todo por HTTP (no IPC). Los hooks lo envuelven en queries.
- Si la lectura de config falla (JSON corrupto → 500), abrir el wizard con el **template**
  igual (no bloquear); ver `ConfigWizardPanel`.
- Tras guardar en el wizard → `queryClient.invalidateQueries` de `['models-list']`,
  `['model-config', name]` y `['models']` (reemplaza el re-escaneo manual).
- Estados vacío/error explícitos.

### ModelDropzone → `useImportModels` *(medio)*

```ts
export function useImportModels() {
  const qc = useQueryClient();
  return useMutation({
    // sube los File UNO A UNO por multipart (POST /models/upload); acumula
    // uploaded/failed por archivo y NO tira (un fallo no aborta el resto).
    mutationFn: async ({ files, onProgress }: { files: File[]; onProgress?: ... }) => { ... },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['models-list'] }),
  });
}
```

Reglas: extensiones aceptadas `.onnx/.tflite/.h5/.keras/.pt/.pth` (filtro UX; la garantía
real — extensión + nombre seguro — la aplica el backend antes de escribir). Ya **no** se
resuelve ningún path (`getPathForFile` eliminado): se suben los `File` del browser tal cual,
con barra de progreso (`onUploadProgress`). Estados `drag-over`/feedback por clases Tailwind.

### ConfigWizard ("wizard-config") → *(profundo)*

Rediseño del monolito `configBuilder.js` (~740 líneas, estado de módulo + `innerHTML` +
`toggleHidden`). El destino lo parte en: **store de estado**, **componentes por paso**,
**campos controlados** y una **función pura de guardado**.

#### Estado → `store/wizardStore.ts`

```ts
interface WizardState {
  modelFile: string;          // ej. "yolov7-tiny.onnx"
  step: 1 | 2 | 3 | 4;
  config: ModelConfig;        // tipo del schema Pydantic
  open(modelFile: string, existing: ModelConfig | null): void;
  close(): void;
  setStep(step: 1 | 2 | 3 | 4): void;
  setField(path: string, value: unknown): void;   // reemplaza setDeep()
  setModelType(type: ModelType): void;            // resetea config.output a los defaults del tipo
}
```

- `setField(path, value)` reemplaza el `setDeep(obj, parts, value)` actual (asignación por
  path tipo `"output.tensor_structure.num_classes"`).
- `setModelType` re-inicializa `config.output` con la plantilla del nuevo tipo.

#### Defaults → `useConfigTemplate` (cierra la divergencia)

**Cambio clave:** los `OUTPUT_DEFAULTS`/`INPUT_DEFAULTS`/`RUNTIME_DEFAULTS`/`ANCHOR_DEFAULTS`
hardcodeados (ver divergencia arriba) **se eliminan**. El wizard pide los defaults al
backend, que ya es la single source of truth:

```ts
// hooks/useConfigTemplate.ts
export function useConfigTemplate(type: ModelType) {
  return useQuery({
    queryKey: ['config-template', type],
    queryFn: () => getConfigTemplate(type),  // GET /config/template/{type}
    staleTime: Infinity,                      // los defaults no cambian en runtime
  });
}
```

Al abrir el wizard para un modelo sin config, se usa la plantilla del backend como
`config` inicial. Para uno con config existente, se carga la suya.

#### Pasos y campos

- Un componente por paso: `Step1Type`, `Step2Input`, `Step3Output/{Detection,Classification,
  Segmentation}Output`, `Step4Runtime`. El paso 3 se elige por `config.model_type`.
- Los helpers `num/numf/sel/chk/txt` se vuelven **componentes controlados**
  (`NumberField`, `FloatField`, `SelectField`, `CheckField`, `TextField`) que leen del
  store y escriben con `setField(path, value)`.
- **Campos condicionales = render declarativo** (sin `toggleHidden`/`innerHTML`): se muestran
  con `{cond && <Section/>}`. Casos: `normalize`→mean/std; `letterbox`→pad color (solo
  detection); `apply_nms`→nms_threshold; `pack_format === 'anchor_deltas'`→sección anchors;
  `backend`→sección ONNX/TFLite; `box_format`→llaves de coordenadas.
- Letterbox y anchors **solo para detection**.

#### Guardado → `toBackendConfig` (puro) + `useSaveConfig`

Las transformaciones del `save()` actual se extraen a una **función pura testeable**:

```ts
// lib/toBackendConfig.ts — alinea el estado del wizard al schema estricto del backend
export function toBackendConfig(state: ModelConfig): ModelConfig {
  const out = structuredClone(state);
  // 1. out_coords_space vive en runtime.runtimeShapes, no en output
  if (out.output?.out_coords_space) {
    out.runtime.runtimeShapes ??= {};
    out.runtime.runtimeShapes.out_coords_space = out.output.out_coords_space;
    delete out.output.out_coords_space;
  }
  // 2. anular el backend no usado
  if (out.runtime.backend !== 'tflite') out.runtime.tflite = null;
  if (out.runtime.backend !== 'onnxruntime') out.runtime.onnx = null;
  // 3. anchor_config solo para anchor_deltas
  if (out.model_type === 'detection' && out.output.pack_format !== 'anchor_deltas') {
    out.output.anchor_config = null;
  }
  return out;
}
```

**Decisión de guardado (tomada):** guardar por **`POST /configs/{name}`** (mutation
`useSaveConfig`), que valida contra `ModelConfig` estricto en el backend antes de escribir
(Fase 3). Esto deja `uncaAPI.writeConfig` sin uso → removible (deuda #17, ver
`plataforma-electron.md`). El `baseName` sale de `modelFile` sin extensión; la validación de
nombre seguro la hace el backend (`[A-Za-z0-9_-]`).

```ts
// hooks/useSaveConfig.ts
export function useSaveConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, config }: { name: string; config: ModelConfig }) =>
      saveConfig(name, toBackendConfig(config)),   // POST /configs/{name}
    onSuccess: () => qc.invalidateQueries({ queryKey: ['models'] }),
  });
}
```

#### Tipado

`ModelConfig` y sus sub-tipos (`input`, `output`, `runtime`, `anchor_config`, …) se derivan
del schema Pydantic estricto (`config_schema.py`) y viven en `shared/api/types.ts`. Así el
wizard no puede armar un config con un campo que el backend va a rechazar.

#### Capacidad crítica: redirección de errores del backend (SDD §1.2)

El wizard **no** valida reglas de ML; su responsabilidad crítica es la **gestión de estados
asíncronos de red**: interceptar el rechazo del backend, decodificarlo y **redirigir al
campo que causó el conflicto**. Implementado en `lib/validationErrors.ts` (puro):

- `fieldIssuesFrom(err)` toma el `ApiError` 422, lee la lista Pydantic de `detail`
  (`{ loc, msg }`) y devuelve `[{ path, msg }]` con el `path` en dotted notation alineado a
  los paths de `setField` (ej: `input.width`).
- `stepOfPath(path)` mapea la raíz del `loc` (`model_type`/`input`/`output`/`runtime`) al
  paso 1–4.
- En `handleSave`, ante un 422: se guardan los issues en `wizardStore.fieldErrors`, se
  **navega automáticamente al primer paso con error** y se muestra un **banner que nombra
  cada campo** (clickable → salta a su paso). Editar un campo limpia su error. Si el 422 no
  trae `loc` mapeable, cae a un mensaje genérico (no rompe). El guardado queda abortado.

> El resaltado inline (borde rojo en cada `input`) se evaluó y se dejó fuera por bajo
> retorno: el banner + la navegación ya cumplen "redirigir al campo específico".

#### Testing

| Qué | Cómo |
|---|---|
| `toBackendConfig` | Función pura: casos detection/classification/segmentation + cada transformación (mover `out_coords_space`, anular backends, anular `anchor_config`). El de mayor valor. |
| `setField`/`setModelType` | Store: asignación por path y reseteo de output al cambiar tipo. |
| Campos condicionales | Render: que las secciones aparezcan/desaparezcan según el campo gatillo. |
| `useConfigTemplate` | Que el wizard arranque con los defaults del backend, no hardcodeados. |
