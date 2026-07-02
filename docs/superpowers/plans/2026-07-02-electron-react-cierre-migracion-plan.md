# Plan de implementacion — Cierre de la migracion: React en Electron + borrado del cliente viejo

**Spec:** `docs/superpowers/specs/2026-07-02-electron-react-cierre-migracion-design.md`
**Fecha:** 2026-07-02
**Regla transversal:** enfoque "verificar primero, borrar despues" — el slice 3 (borrado)
NO arranca hasta que el slice 2 (verificacion en Electron dev + prod) este verde en los
4 flujos. Comentarios en espanol sin tildes (convencion del repo). No se toca Python.
Se puede commitear por slice.

---

## Slice 0 — Portar el ciclo de vida del backend (backend-process.js)

**Depende de:** nada. **Bloquea:** slice 2 (la verificacion valida este wiring).

1. Copiar el modulo desde la otra rama, tal cual (NO cherry-pick del commit 5219162,
   que mezcla cambios de metricas del backend):

   ```bash
   git show refactor-agente-fase1:src/backend-process.js > src/backend-process.js
   ```

2. Wiring en `src/main.js`, replicando el de la otra rama (consultar
   `git show refactor-agente-fase1:src/main.js` como referencia):
   - `const { startBackend, stopBackend } = require('./backend-process');`
   - Raiz del repo: `path.join(__dirname, '..')` (main.js vive en `src/`).
   - En `app.whenReady()`: `startBackend({ projectRoot })` ANTES de `createWindow()`,
     sin esperar a que el backend responda (el frontend tolera backend ausente y
     reintenta el WS). Aplica en dev y en prod por igual.
   - En `app.on('will-quit')`: `stopBackend()`.

**Verificacion:**
- `npm run start:dev` SIN uvicorn corriendo: la app abre y `GET http://127.0.0.1:8000/get_models`
  responde (backend levantado por Electron). Al cerrar la ventana, el proceso de uvicorn
  muere (chequear con `Get-Process | Where-Object ProcessName -match python`).
- `UNCA_NO_SPAWN=1` + uvicorn a mano: la app funciona igual y NO lanza un segundo proceso.

---

## Slice 1 — Rescate del logo como favicon

**Depende de:** nada.

1. Copiar `static/images/logo5.svg` -> `client/src/assets/logo.svg` (copiar, no mover:
   el arbol viejo queda intacto hasta el slice 3, donde se borra el original).
2. `client/index.html`: agregar `<link rel="icon" type="image/svg+xml" href="/src/assets/logo.svg" />`
   en el `<head>`. Vite resuelve la ruta en dev y la fingerprintea en el build (verificar
   que el `<link>` del `dist/index.html` sale con ruta relativa `./assets/...` por el
   `base: './'`).
3. El icono de ventana (.ico) queda FUERA (pendiente menor, ver spec §4).

**Verificacion:** `npm run typecheck` + `npm run build`; favicon visible en el tab en dev.

---

## Slice 2 — Verificacion en Electron real (dev y prod)

**Depende de:** slices 0-1. **Bloquea:** slice 3.

1. **Harness CDP** (en el scratchpad de la sesion, NO en el repo): script Node que se
   conecta a `http://127.0.0.1:9222/json`, abre el WebSocket de la page y usa
   `Page.captureScreenshot` + `Runtime.evaluate` para screenshots e interaccion.
   Electron se lanza con el puerto de debug:

   ```bash
   npx electron . --dev --remote-debugging-port=9222   # modo dev
   npx electron . --remote-debugging-port=9222         # modo prod (tras npm run build)
   ```

   Fallback si CDP no coopera con la ventana: checklist manual con el usuario.

2. **Flujos a verificar, en AMBOS modos** (dev con dev-server Vite; prod con
   `client/dist` bajo `file://`):
   1. La app abre; las vistas renderizan con la piel Cabina Tecnica; las fuentes
      locales (Space Grotesk / JetBrains Mono) cargan — sensibles al `base: './'`
      bajo `file://`.
   2. Cargar `yolov7-tiny` desde el selector de modelos (POST /select_model OK).
   3. Inferencia sobre archivo (imagen o video): frames por WS, cajas dibujadas en el
      canvas. Camara en vivo FUERA de alcance (se anota como no cubierta).
   4. Wizard round-trip: abrir la config de un modelo existente, recorrer los 4 pasos,
      guardar; el JSON resultante valida contra el schema estricto (POST /configs sin
      422) y el modelo sigue cargando. Cierra los pendientes de verificacion visual
      del wizard simplificado y del re-skin.

3. Guardar las capturas de cada flujo (scratchpad) y dejar constancia del resultado en
   el mensaje de cierre del slice. Cualquier bug encontrado se arregla ANTES de pasar
   al slice 3 (el cliente viejo sigue en el arbol como referencia).

**Verificacion:** los 4 flujos verdes en dev Y prod.

---

## Slice 3 — Borrado del cliente viejo

**Depende de:** slice 2 verde.

1. `git rm -r src/render static` (se lleva `scripts.js` + los 10 modulos, `index.html`,
   `styles.css` y los 3 logos; `logo5.svg` ya vive copiado en `client/src/assets/`).
2. Limpiar comentarios muertos:
   - `src/main.js` cabecera: quitar "El frontend viejo (static/index.html + src/render)
     queda sin uso y se puede borrar".
   - `vite.config.mts` cabecera: quitar "convive con la app vieja (src/render + static/)
     SIN tocarla" y la nota "El switch a este build se hace recien al final de la
     migracion" (ya ocurrio).
3. Chequeo de referencias vivas: `grep -r "render/" "static/" "uncaAPI"` sin hits en
   codigo (`src/`, `client/`, configs, package.json, vite.config). Los hits en docs
   historicos (`docs/superpowers/`, `Informacion/`) son aceptables.

**Verificacion:** `npm run typecheck` + `npm run build` verdes; `pytest` intacto;
`npm run start:dev` sigue abriendo la app (smoke rapido post-borrado).

---

## Slice 4 — Documentacion

**Depende de:** slice 3.

1. `CLAUDE.md`:
   - Seccion 1: quitar "El frontend viejo (src/render/ + static/) quedo muerto/removible";
     mencionar que Electron arranca/mata uvicorn (backend-process.js) con escape
     `UNCA_NO_SPAWN=1`.
   - Seccion 2 (como correr): con Electron ya no hace falta la terminal de uvicorn
     (queda documentado el flujo manual con `UNCA_NO_SPAWN=1` para desarrollo del backend).
   - Seccion 3 (mapa): eliminar el arbol `src/render/` + `static/` y la nota que los
     marca como muertos; agregar `backend-process.js`.
   - Pendientes: #17 cerrado (preload/ipc-handlers ya eran no-op; el borrado del arbol
     viejo elimina el resto), #18 cerrado (verificacion del slice 2). Anotar pendiente
     menor nuevo: icono de ventana .ico para Electron en Windows.
   - Actualizar la nota de "Ultima actualizacion" de cabecera.
2. `docs/frontend-components/plataforma-electron.md`: Electron arranca el backend;
   el cliente viejo no existe mas. Revisar `docs/frontend-components/README.md` por
   referencias al arbol viejo y ajustar si las hay.
3. Actualizar la memoria persistente si corresponde (estado del bloque de reformas).

**Verificacion:** lectura cruzada CLAUDE.md <-> arbol real (que el mapa no mienta).

---

## Riesgos y mitigacion

- **Prod bajo `file://` nunca probado:** assets/fuentes con rutas relativas. Mitigado por
  `base: './'` ya configurado y porque el slice 2 corre ANTES del borrado.
- **Huerfanos de uvicorn en Windows:** el modulo maneja el cierre normal (`will-quit`);
  si Electron muere abruptamente puede quedar un proceso python vivo — caso patologico
  aceptado (spec §5). La verificacion del slice 0 chequea el cierre normal.
- **CDP contra Electron:** si `--remote-debugging-port` no permite manejar la app,
  fallback a checklist manual con el usuario (spec §2.3).
- **Doble backend en desarrollo:** si el usuario tiene uvicorn a mano y Electron lanza
  otro, el segundo falla por puerto ocupado (uvicorn no arranca, el frontend usa el
  existente) — comportamiento aceptable; `UNCA_NO_SPAWN=1` es el escape limpio.
