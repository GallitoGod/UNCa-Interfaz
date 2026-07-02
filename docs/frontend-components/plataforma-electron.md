# Plataforma: Electron (main process)

La capa nativa: creación de la ventana con hardening de seguridad y el puente seguro
`contextBridge`. La app se mantiene en Electron como contenedor nativo.

Archivos as-is: `src/main.js`, `src/preload.js`, `src/ipc-handlers.js`,
`src/backend-process.js`.

> **Estado (2026-07-02): cierre de la migración.** El cliente viejo (`src/render/` +
> `static/`) **se eliminó del árbol** (queda en el historial de git). Además el main
> process ahora **arranca y detiene uvicorn** vía `src/backend-process.js` (portado de la
> rama `refactor-agente-fase1`): `startBackend()` en `app.whenReady()` (no bloquea la
> ventana; el frontend reintenta sus queries mientras el backend bootea) y `stopBackend()`
> en `before-quit`. Escapes: `UNCA_NO_SPAWN=1` (backend a mano) y `UNCA_PYTHON` (forzar
> intérprete). Verificado en Electron real (dev y prod `file://`): render + fuentes,
> carga de modelo, inferencia sobre archivo y round-trip del wizard.

> **🔴 Estado (2026-06-26): cambio de diseño grande.** El SDD impone un **thin client SIN
> acceso a disco desde Electron** (§2, §1.2). En consecuencia **se eliminó todo el IPC de
> disco** que esta doc describía como contrato central:
> - `window.uncaAPI` (listModels/readConfig/importModels/writeConfig/getPathForFile) **ya no
>   existe**: se borró `client/src/shared/electron/uncaApi.ts`.
> - Los handlers `models:list/import` y `configs:read/write` se eliminaron de
>   `ipc-handlers.js` → `registerIpcHandlers()` quedó **no-op** (seam para IPC futuro).
> - `preload.js` quedó **sin exposiciones** (contextBridge vacío).
> - Toda la persistencia pasó al backend HTTP: `GET /models`, `GET/POST /configs/{name}`,
>   `POST /models/upload` (ver `feature-modelos.md`).
> - `main.js` ahora carga el **build de Vite** (`client/dist/index.html` en prod; dev server
>   con `electron . --dev`) en vez de `static/index.html`.
>
> Las secciones *as-is* de abajo describen el contrato IPC **histórico (ya removido)**; se
> conservan como registro. La convención `{ success, data?, error? }` del SDD §4.1.1 queda
> documentada para cualquier IPC **futuro no-disco** (hoy no hay ninguno).

---

## MainProcess/Window

- **Responsabilidad:** Crear la `BrowserWindow` con la configuración de seguridad
  recomendada, registrar los handlers IPC y cargar `static/index.html`.
- **Entradas:** ciclo de vida de Electron (`app.whenReady`, `activate`,
  `window-all-closed`).
- **Salidas:** la ventana principal; registro de handlers IPC; carga del HTML.
- **Dependencias:** `electron` (`app`, `BrowserWindow`), `ipc-handlers.registerIpcHandlers`,
  `preload.js`.
- **Reglas de negocio (hardening — no relajar):**
  - `nodeIntegration: false` → el renderer no tiene `require`/`fs`/`process`.
  - `contextIsolation: true` → el preload corre en un mundo JS separado; solo lo expuesto
    por `contextBridge` llega a la página.
  - `sandbox: true` → renderer sandboxeado a nivel OS.
  - Los handlers IPC se registran **antes** de crear la ventana (el renderer nunca invoca
    un canal inexistente).
  - En no-macOS, la app sale al cerrar todas las ventanas.
- **Mapeo al destino React:** `electron/main.ts`. La única diferencia con React+Vite:
  `loadFile('static/index.html')` pasa a cargar el build de Vite (`loadURL` al dev server
  en desarrollo, `loadFile(dist/index.html)` en producción). El hardening se mantiene
  idéntico. (Nota 2026-07-02: el `backend-process.js` que arranca/mata uvicorn desde el
  main **ya existe** — ver la nota de estado de arriba.)

## PreloadBridge (`uncaAPI`)

- **Responsabilidad:** Exponer al renderer aislado, vía `contextBridge`, **solo** las
  operaciones permitidas, como funciones async que delegan en el main por IPC.
- **Entradas:** llamadas del renderer a `window.uncaAPI.*`.
- **Salidas:** `ipcRenderer.invoke(canal, ...args)` hacia los handlers; expone
  `webUtils.getPathForFile`.
- **API expuesta (`window.uncaAPI`):**
  | Método | Canal IPC | Devuelve |
  |---|---|---|
  | `listModels()` | `models:list` | `{ ok, models:[{file,ext,baseName,hasConfig}] }` |
  | `importModels(paths)` | `models:import` | `{ ok, copied, errors:[{file,error}] }` |
  | `readConfig(baseName)` | `configs:read` | `{ ok, config }` (config `null` si no existe) |
  | `writeConfig(baseName, config)` | `configs:write` | `{ ok }` o `{ ok:false, error }` |
  | `getPathForFile(file)` | — (directo) | path absoluto del `File` arrastrado |
- **Dependencias:** `electron` (`contextBridge`, `ipcRenderer`, `webUtils`).
- **Reglas de negocio:**
  - **Regla de oro:** acá no se implementa lógica de archivos; solo se reenvía al main.
  - `getPathForFile` es el reemplazo oficial de `File.path` (eliminado en Electron ≥ 32) y
    **solo** está disponible en el preload.
- **Mapeo al destino React:** `electron/preload.ts` (sin cambios de fondo). En el lado
  React se tipa `window.uncaAPI` (`shared/api/types.ts` o `electron.d.ts`) y se envuelve en
  servicios (`features/models/api/*`) consumidos por hooks de TanStack Query. **Pendiente
  de limpieza:** `writeConfig` quedaría sin uso si el guardado del wizard migra a
  `POST /configs` (deuda #17 en CLAUDE.md) — depende de la decisión de guardado.

## IpcFileHandlers

- **Responsabilidad:** Implementar en el main process **todas** las operaciones de `fs`
  del frontend, validando entradas. Cada handler devuelve `{ ok, ... }` en vez de tirar.
- **Entradas:** invocaciones IPC: `models:list`, `models:import`, `configs:read`,
  `configs:write`.
- **Salidas:** lectura/escritura/copia en `models/` y `configs/`; objetos resultado.
- **Dependencias:** `electron` (`ipcMain`, `app`), `fs`, `path`.
- **Reglas de negocio (seguridad):**
  - Raíz determinística: `ROOT_DIR = app.getAppPath()` (no depende del cwd). De ahí
    `MODELS_DIR` y `CONFIGS_DIR`.
  - **Anti path-traversal** (`isSafeBaseName`): el `baseName` no puede tener `/`, `\`,
    `..`, ni ser `.`; longitud `[1, 256)`. Se aplica en `configs:read` y `configs:write`.
  - **Revalidación de extensión** en `models:import` (el filtro del renderer es solo UX);
    set `SUPPORTED_EXTENSIONS` debe mantenerse sincronizado con `MODEL_EXTENSIONS` en
    `mainAPI.py` y con el del renderer.
  - Los errores cruzan el IPC **como datos** (`{ ok:false, error }`), no como excepciones:
    la UI decide cómo mostrarlos.
  - `configs:read` de un archivo inexistente devuelve `{ ok:true, config:null }` (no es
    error: habilita defaults en el wizard).
- **Mapeo al destino React:** `electron/ipc-handlers.ts` (sin cambios de fondo). Es la
  frontera de confianza; **no** mover esta lógica al renderer/React bajo ningún concepto.
  Se recomienda tipar los contratos de cada canal y compartir los tipos con el front.

---

## Frontera de confianza (resumen)

```
Renderer (React, sandbox)
        │  window.uncaAPI.*  (solo lo expuesto)
        ▼
PreloadBridge (contextIsolation)
        │  ipcRenderer.invoke(canal)
        ▼
IpcFileHandlers (main process)  ──>  fs sobre models/ y configs/
        ▲  valida baseName + extensión, raíz = app.getAppPath()
```

Todo `fs` del frontend cruza estos tres saltos. Mantener esa propiedad en la migración:
React **nunca** toca disco directamente.

---

## Diseño to-be (React + Electron) — implementado

> Lo realmente construido difiere del plan original (que asumía "IPC sobrevive"). El SDD
> mandó thin client sin disco, así que esta capa **adelgazó**: Electron quedó como puro
> contenedor de ventana. Cambios efectivos: `main.js` carga el build de Vite; el IPC de
> disco se **eliminó** (handlers no-op, preload vacío, `uncaApi.ts` borrado). Los 3 archivos
> siguen en `src/` como **JS (CommonJS)** — no se migraron a TS ni a una carpeta `electron/`.

### Estructura (real)

```
src/
├── main.js             # carga client/dist (prod) | dev server (electron . --dev); arranca uvicorn
├── preload.js          # contextBridge VACIO (sin API de disco)
├── ipc-handlers.js     # registerIpcHandlers() no-op (sin handlers de disco)
└── backend-process.js  # startBackend/stopBackend: ciclo de vida de uvicorn (2026-07-02)
```

### MainProcess → `electron/main.ts` *(liviano)*

Único cambio real: el origen del HTML pasa a depender del modo.

```ts
if (import.meta.env?.DEV || process.env.NODE_ENV === 'development') {
  mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL!); // dev server de Vite
} else {
  mainWindow.loadFile(path.join(__dirname, '../dist/index.html')); // build de Vite
}
```

El hardening (`nodeIntegration:false`, `contextIsolation:true`, `sandbox:true`) y el
registro de handlers antes de crear la ventana **no cambian**. (Nota 2026-07-02: el
`backend-process.js` que arranca/mata uvicorn **ya existe y está cableado** en `main.js` —
ver la nota de estado del principio.)

### PreloadBridge → `electron/preload.ts` + `uncaAPI.d.ts` *(liviano)*

El preload no cambia de fondo; se agrega el tipo para que el renderer tenga autocompletado
y chequeo:

```ts
// src/shared/electron/uncaAPI.d.ts
export interface ModelEntry { file: string; ext: string; baseName: string; hasConfig: boolean; }

export interface UncaAPI {
  listModels(): Promise<{ ok: boolean; models: ModelEntry[]; error?: string }>;
  importModels(paths: string[]): Promise<{ ok: boolean; copied: number; errors: { file: string; error: string }[] }>;
  readConfig(baseName: string): Promise<{ ok: boolean; config: ModelConfig | null; error?: string }>;
  writeConfig(baseName: string, config: ModelConfig): Promise<{ ok: boolean; error?: string }>;
  getPathForFile(file: File): string;
}

declare global {
  interface Window { uncaAPI: UncaAPI; }
}
```

Los servicios de cada feature (`features/models/api/*`) envuelven `window.uncaAPI` y los
hooks de TanStack Query los consumen. `ModelConfig` es el tipo del schema Pydantic (ver
`feature-modelos.md`, to-be).

### IpcFileHandlers → `electron/ipc-handlers.ts` *(liviano)*

Sin cambios de fondo: **es la frontera de confianza**, no se mueve al renderer. Mejora
sugerida: tipar el payload de cada canal y compartir los tipos con el front (mismo
`uncaAPI.d.ts`). El set `SUPPORTED_EXTENSIONS` sigue debiendo estar sincronizado con
`MODEL_EXTENSIONS` (backend) y el del renderer.

### Decisión ~~abierta~~ RESUELTA: sin disco por IPC

La decisión se cerró a favor de **sin disco**: no sólo `writeConfig`, **todos** los canales
de disco (`models:list/import`, `configs:read/write`) y `getPathForFile` se eliminaron. El
guardado del wizard va por `POST /configs/{name}`; el resto, por los endpoints HTTP
correspondientes. `uncaApi.ts` borrado; `ipc-handlers.js`/`preload.js` no-op. La frontera de
confianza ya no vive en el main process de Electron sino en el **backend FastAPI**, que valida
nombre seguro (`[A-Za-z0-9_-]`) y extensión antes de escribir.
