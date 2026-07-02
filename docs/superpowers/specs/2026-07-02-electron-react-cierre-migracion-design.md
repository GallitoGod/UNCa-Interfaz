# Cierre de la migración del frontend: React en Electron + borrado del cliente viejo

- **Fecha**: 2026-07-02
- **Rama**: `refactor-frontend-react`
- **Estado**: aprobado (diseño validado por secciones en sesión de brainstorming)

## 1. Contexto y objetivo

La migración a React (2026-06-26) dejó el cableado Electron→React ya hecho:
`src/main.js` carga el dev-server de Vite con `--dev` y el build de `client/dist`
en producción; `preload.js` e `ipc-handlers.js` son no-op (thin client sin disco).
Pero el bloque nunca se cerró:

- **Nunca se corrió Electron de verdad** con el cliente React (pendiente #18 del
  CLAUDE.md, más las notas "falta verificación visual" del wizard simplificado y
  del re-skin Cabina Técnica).
- El **cliente viejo** en JS vanilla (`src/render/` con `scripts.js` + 10
  módulos, más `static/`) sigue en el árbol, muerto.
- El **ciclo de vida del backend** (Fase 4 tarea 1: Electron arranca/mata uvicorn,
  `src/backend-process.js`) quedó en la rama `refactor-agente-fase1` y nunca llegó
  a esta rama: acá el backend se levanta a mano.

**Objetivo**: cerrar el bloque de reformas de la migración. Al terminar, `npm start`
abre la app completa (backend incluido), los flujos clave están verificados en
Electron real, y el cliente viejo no existe más.

## 2. Alcance

Enfoque elegido: **verificar primero, borrar después**. El borrado del cliente
viejo se hace recién cuando la verificación en Electron está verde, para tener el
código viejo como referencia inmediata si aparece un comportamiento no cubierto.

### 2.1 Ciclo de vida del backend en Electron

- Copiar `src/backend-process.js` **tal cual** desde `refactor-agente-fase1`
  (`git show refactor-agente-fase1:src/backend-process.js`). NO cherry-pick: el
  commit 5219162 mezcla el snapshot de métricas del backend, que no corresponde a
  esta rama.
- El módulo ya cubre lo necesario:
  - Resolución del intérprete: `UNCA_PYTHON` > `.venv/Scripts/python.exe` (existe
    en el repo) > `python` del PATH.
  - `startBackend()`: lanza `uvicorn api.mainAPI:app --host 127.0.0.1 --port 8000
    --app-dir src` con cwd en la raíz del repo; idempotente.
  - `stopBackend()`: mata el proceso al salir.
  - Escape `UNCA_NO_SPAWN=1`: no arranca uvicorn (para desarrollo con backend a
    mano en otra terminal — el flujo actual sigue funcionando).
- Wiring en `src/main.js`, replicando el de la otra rama: `startBackend()` en
  `app.whenReady()` sin bloquear la creación de la ventana (el frontend tolera
  backend ausente y reintenta el WS), `stopBackend()` en `will-quit`. Aplica en
  dev y en prod.
- No se toca nada del backend Python ni de `client/` para esta parte.

### 2.2 Rescate del logo

- Mover `static/images/logo5.svg` → `client/src/assets/logo.svg`.
- Usarlo como favicon en `client/index.html`
  (`<link rel="icon" type="image/svg+xml" ...>`, soportado nativo por Chromium).
- El **ícono de ventana** de Electron en Windows requiere `.ico`/`.png` (no acepta
  SVG): queda FUERA de este bloque, anotado como pendiente menor. La ventana usa
  el ícono default de Electron por ahora.
- `logo3.svg` y `logo4.svg` no se rescatan (quedan en el historial de git).

### 2.3 Verificación en Electron real (antes de borrar)

Dos modos:

- **Dev**: `npm run dev` + `npm run start:dev` (Electron → dev-server Vite).
  Valida también 2.1: el backend lo arranca Electron solo.
- **Prod**: `npm run build` + `npm start` (Electron → `client/dist` bajo
  `file://`). Modo nunca probado: rutas relativas de assets (`base: './'`),
  fuentes empaquetadas locales, WS a `127.0.0.1:8000`.

Flujos a verificar en ambos modos:

1. La app abre y las vistas renderizan con la piel Cabina Técnica, fuentes
   locales incluidas.
2. Cargar un modelo real (YOLOv7) desde el selector.
3. Inferencia sobre **archivo** (video o imagen): frames por WS, cajas dibujadas
   en el canvas.
4. Wizard round-trip: abrir la config de un modelo existente, recorrer los 4
   pasos, guardar; el JSON resultante valida contra el schema estricto y el
   modelo sigue cargando. Cierra los pendientes de verificación visual del
   wizard simplificado y del re-skin.

Mecánica: Electron lanzado con `--remote-debugging-port` y manejado por CDP
(screenshots + interacción). Fallback si CDP no coopera con la ventana de
Electron: pasada visual del usuario con checklist corta. La **cámara en vivo
queda fuera** (requiere hardware/participación del usuario) y se anota como no
cubierta.

### 2.4 Borrado del cliente viejo (después de verificar)

- Eliminar `src/render/` completo: `scripts.js` + `modules/` (constants, overlay,
  streamHandler, cameraSwitcher, fileHandler, record, modelLoader, selectModel,
  modelsManager, configBuilder).
- Eliminar `static/` completo: `index.html`, `styles.css`, `images/`.
- Limpiar referencias muertas en comentarios: cabeceras de `src/main.js` ("el
  frontend viejo queda sin uso y se puede borrar") y `vite.config.mts` ("convive
  con la app vieja SIN tocarla").
- Verificación post-borrado: `grep` de `render/`, `static/` y `uncaAPI` sin
  referencias vivas en código (docs históricos/specs pueden mencionarlos).

### 2.5 Documentación

- `CLAUDE.md`:
  - Sección 1: quitar "frontend viejo muerto/removible".
  - Sección 2 (cómo correr): ya no hace falta la terminal de uvicorn con
    Electron; nota del escape `UNCA_NO_SPAWN=1` para el flujo manual.
  - Sección 3 (mapa): desaparece el árbol `src/render/` + `static/`; entra
    `backend-process.js`.
  - Pendientes: #17 (handler IPC `writeConfig`) cerrado de facto — `preload.js`
    e `ipc-handlers.js` ya son no-op; #18 (verificación visual) cerrado por 2.3.
  - Nota de última actualización.
- `docs/frontend-components/plataforma-electron.md` (+ README de la carpeta si
  corresponde): Electron arranca el backend; el cliente viejo no existe más.

## 3. Criterios de éxito

- `npm start` (sin nada más corriendo) abre la app completa y funcional,
  backend incluido.
- Flujos 1–4 de 2.3 verificados en dev y prod, con capturas.
- `src/render/` y `static/` eliminados; sin referencias vivas en código.
- `npm run typecheck` y `npm run build` verdes; `pytest` intacto (no se toca
  Python).

## 4. Fuera de alcance

- Instalador / electron-builder / Python embebido (proyecto aparte).
- Ícono de ventana `.ico` para Electron en Windows (pendiente menor).
- Cámara en vivo en la verificación.
- Pipelines reales de CLS/SEG (pendiente #7, backend).
- Reestructura de carpetas (mover Electron a `electron/`, separar
  `client/package.json`).

## 5. Riesgos conocidos

- **Prod bajo `file://`**: primera corrida real; posibles sorpresas con rutas de
  assets o fuentes. Mitigado por `base: './'` ya configurado y por verificar
  antes de borrar.
- **Spawn de uvicorn en Windows**: matar el proceso al salir puede dejar
  huérfanos si Electron muere abruptamente; el módulo de la otra rama ya maneja
  el cierre normal (`will-quit`), y el caso patológico se acepta.
- **CDP contra Electron**: si `--remote-debugging-port` no permite manejar la
  app, se cae al fallback manual (checklist con el usuario).
