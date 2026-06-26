# Frontend de UNCaLens — Componentes (as-is → to-be)

Inventario y documentación del frontend **actual** (Electron + JS vanilla) descompuesto
en componentes, con el **mapeo al destino** de la nueva arquitectura.

- **Origen (as-is):** Electron con renderer aislado (`contextIsolation`, `sandbox`),
  JS vanilla por módulos ES. Acceso a disco en el main process detrás de IPC.
- **Destino (to-be):** **Thin Client** con **Feature-Driven Architecture** sobre
  **React + Vite + TypeScript + Zustand + TanStack Query + Axios + Tailwind**,
  **seguir empaquetado en Electron** (la capa main/preload/IPC sobrevive).

> Regla de oro de estos docs: describen lo que el código **hace hoy**, no lo que otra
> documentación afirma. Donde el código y `CLAUDE.md` divergen, se documenta el código
> y se marca la divergencia (ver `feature-modelos.md`, ConfigWizard).

> **✅ Estado (2026-06-26): el to-be está IMPLEMENTADO** en `client/` (rama
> `refactor-frontend-react`, sin commitear). El loop guiado por el SDD
> (`Informacion/UNCa-Lens-SDD.md`) cerró los 5 componentes. Las secciones *as-is* de cada
> doc describen el frontend viejo (`src/render` + `static/`), hoy **muerto/removible**. Hubo
> **dos cambios de decisión** respecto al to-be original de estos docs:
> 1. **Thin client SIN disco:** el SDD prohíbe el acceso a disco desde Electron. **El IPC de
>    disco se eliminó** — listar/leer/escribir/importar modelos y configs van por HTTP al
>    backend (`GET /models`, `GET/POST /configs/{name}`, `POST /models/upload`). Donde abajo
>    diga "IPC sobrevive", leer este punto.
> 2. **El stream se pausa/reanuda** al navegar Inferencia↔Modelos (antes era "mejora
>    opcional"). Ver `app-shell.md`.

---

## Cómo leer cada documento

Cada componente se documenta con seis campos fijos:

1. **Responsabilidad** — una frase: qué hace y qué NO hace.
2. **Entradas** — props/eventos DOM, datos, endpoints consumidos, APIs del browser.
3. **Salidas** — eventos emitidos, efectos secundarios, archivos generados, llamadas al backend.
4. **Dependencias** — otros componentes/módulos, endpoints, librerías del browser.
5. **Reglas de negocio** — invariantes que NO se pueden romper sin regresiones.
6. **Mapeo al destino React** — carpeta feature + si va como componente / store Zustand /
   hook TanStack Query / servicio Axios, con notas de migración.

---

## Mapa de features

| Documento | Vista / capa | Componentes |
|---|---|---|
| [`app-shell.md`](./app-shell.md) | Chrome común | Bootstrap · ViewRouter · SourceTabs · ThemeToggle · BackendClient |
| [`feature-inferencia.md`](./feature-inferencia.md) | Vista "Inferencia" | VideoStreamClient · CameraSource · FileSource · DetectionOverlay · Recorder · ModelSelector · ConfidenceControl · MetricsHUD · InferenceLogPanel · DrawSettingsModal |
| [`vision-workspace.md`](./vision-workspace.md) | Diseño (to-be) del canvas | VisionWorkspace · estrategias por tipo (detection/classification/segmentation) · registry |
| [`feature-modelos.md`](./feature-modelos.md) | Vista "Modelos" | ModelsManager · ModelDropzone · ConfigWizard (el "wizard-config") |
| [`plataforma-electron.md`](./plataforma-electron.md) | Main process | MainProcess/Window · PreloadBridge (`uncaAPI`) · IpcFileHandlers |

## Mapa as-is → to-be (archivos)

| Archivo actual | Componente(s) | Doc |
|---|---|---|
| `static/index.html` | markup base de todas las vistas | app-shell |
| `src/render/scripts.js` | Bootstrap, ViewRouter, SourceTabs, ThemeToggle + wiring de controles de inferencia | app-shell + feature-inferencia |
| `src/render/modules/constants.js` | BackendClient (URLs) | app-shell |
| `src/render/modules/streamHandler.js` | VideoStreamClient | feature-inferencia |
| `src/render/modules/cameraSwitcher.js` | CameraSource | feature-inferencia |
| `src/render/modules/fileHandler.js` | FileSource | feature-inferencia |
| `src/render/modules/overlay.js` | DetectionOverlay + drawSettings | feature-inferencia |
| `src/render/modules/record.js` | Recorder | feature-inferencia |
| `src/render/modules/modelLoader.js` + `selectModel.js` | ModelSelector | feature-inferencia |
| `src/render/modules/modelsManager.js` | ModelsManager + ModelDropzone | feature-modelos |
| `src/render/modules/configBuilder.js` | ConfigWizard ("wizard-config") | feature-modelos |
| `src/main.js` | MainProcess/Window | plataforma-electron |
| `src/preload.js` | PreloadBridge (`uncaAPI`) | plataforma-electron |
| `src/ipc-handlers.js` | IpcFileHandlers | plataforma-electron |

## Estructura de carpetas propuesta (destino)

```
src/
├── app/                       # Bootstrap React, router de vistas, providers (Query, theme)
│   ├── App.tsx
│   ├── router.tsx             # ViewRouter (Inferencia | Modelos)
│   └── providers/
├── shared/                    # Cross-cutting: cliente HTTP, tipos, UI base
│   ├── api/axios.ts           # instancia Axios (baseURL desde constants/env)
│   ├── api/types.ts           # tipos generados/derivados del schema Pydantic
│   └── ui/                    # Tabs, ThemeToggle, Modal, etc. (Tailwind)
├── features/
│   ├── inference/
│   │   ├── components/        # VideoCanvas, MetricsHUD, LogPanel, ConfidenceSlider, DrawSettingsModal
│   │   ├── services/          # videoStream.ts (WebSocket), overlay.ts (canvas imperativo)
│   │   ├── hooks/             # useMetrics, useInferenceLogs (TanStack Query)
│   │   ├── store/             # streamStore (Zustand): fuente activa, mirror, estado WS
│   │   └── api/               # models.ts (get_models, select_model), confidence.ts
│   └── models/
│       ├── components/        # ModelsGrid, ModelCard, ModelDropzone, ConfigWizard/*
│       ├── hooks/             # useModelsList (REST), useConfigTemplate (TanStack Query)
│       ├── store/             # wizardStore (Zustand): estado del wizard de 4 pasos
│       └── api/               # models.ts (GET /models, POST /models/upload), configs.ts (GET/POST)
└── electron/                  # main.js, preload.js, ipc-handlers.js (sin React)
```

> **Nota de implementación:** el código React vive en `client/src/` (no en `src/`, que es
> compartido con el backend Python + Electron). Los 3 archivos de Electron siguen en `src/`
> como **JS (CommonJS)**, no migraron a TS/`electron/`. Y `api/models.ts`/`configs.ts` son
> **REST** (sin `modelsImport.ts` por IPC: ese contrato se eliminó).

## Decisiones de stack que atraviesan el mapeo

- **REST → TanStack Query + Axios.** `/get_models`, `/select_model`, `/config/*`,
  `/metrics`, `/logs/inference`. Métricas y logs usan `refetchInterval` (reemplazan los
  `setInterval` manuales actuales).
- **WebSocket de video → servicio dedicado + Zustand.** NO va por TanStack Query: es un
  stream binario long-lived, no request/response.
- **Canvas (overlay) imperativo.** Componente con `ref`; **no** re-render de React por
  frame (mataría el FPS). Es una regla de performance, no un detalle.
- **`GET /config/template/{model_type}` como single source of truth.** El wizard debe
  consumirlo y tiparlo en vez de re-hardcodear defaults (ver la divergencia documentada
  en `feature-modelos.md`).
- **~~IPC sigue vivo~~ → REVERTIDO: sin disco vía Electron.** Por mandato del SDD (thin
  client), importar modelos y leer/escribir configs **dejaron de pasar por IPC**: van por HTTP
  al backend (`GET /models`, `GET/POST /configs/{name}`, `POST /models/upload`). `uncaApi.ts`
  se borró y los handlers de disco del main process quedaron no-op. Efecto colateral: la vista
  Modelos funciona también en un browser de dev (sin Electron). Ver `plataforma-electron.md`.
