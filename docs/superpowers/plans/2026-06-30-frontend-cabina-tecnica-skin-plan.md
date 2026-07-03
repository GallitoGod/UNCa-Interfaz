# Plan de implementacion — Re-skin "Cabina Tecnica"

**Spec:** `docs/superpowers/specs/2026-06-30-frontend-cabina-tecnica-skin-design.md`
**Fecha:** 2026-06-30
**Regla transversal:** no cambiar logica (hooks, stores salvo quitar tema, contratos
WS/HTTP, pipeline). Comentarios/docstrings en espanol sin tildes. Tras cada slice:
`npm run typecheck` + `npm run build` deben pasar.

Orden por dependencia: cada slice se apoya en el anterior. Se puede commitear por slice.

---

## Slice 0 — Fuentes + tokens (fundacion)

**Depende de:** nada. **Bloquea:** todo lo demas (define el lenguaje visual).

1. **Extraer fuentes** del manifest del spec
   (`docs/UNCaLens_Spec_Visual.html`): los `.woff2` de Space Grotesk (400/500/600/700)
   y JetBrains Mono (400/500/600/700) estan embebidos en base64 en el
   `<script type="__bundler/manifest">`. Subset latin alcanza para la UI en espanol.
   Decodificar a `client/src/assets/fonts/*.woff2` (script node de un solo uso en
   scratchpad; los UUID de las familias salen de los `@font-face` del template ya
   decodificado).
2. **Reescribir `client/src/index.css`** (Seccion 1 del spec):
   - `@font-face` por familia/peso apuntando a los `.woff2` locales,
     `font-display: swap`. Import via `?url` de Vite o ruta relativa desde el css.
   - `--font-sans: 'Space Grotesk', system-ui, sans-serif;`
     `--font-mono: 'JetBrains Mono', ui-monospace, monospace;`
   - Reemplazar los `--c-*` por la paleta del spec (tabla §4 del spec). Agregar
     `--c-feed`, `--c-accent-soft`, `--c-label`.
   - Ampliar `@theme` con `--color-feed` y `--color-accent-soft`.
   - Radios: `--radius-sm: 6px`, `--radius-md: 10px`, `--radius-lg: 13px`; control 8px
     como arbitrary value donde haga falta.
   - Eliminar el bloque `:root` claro y `@custom-variant dark`.
   - Utilidades `.lbl` y `.swatch-meta`; `@keyframes recpulse`; reglas del thumb del
     slider (`::-webkit-slider-thumb`, `::-moz-range-thumb`) y de la pista.
   - `body` usa `--color-canvas` / `--color-fg` (ya lo hace).

**Verificacion:** `npm run build`; arrancar dev y confirmar fuentes cargando + fondo
casi-negro.

---

## Slice 1 — Primitivos (`shared/ui`)

**Depende de:** Slice 0.

1. **`Tabs.tsx`** -> look segmentado (pista `bg-control` p-[3px]; activo
   `bg-accent text-accent-fg`; inactivo `text-fg-subtle hover:text-fg`). Conservar
   roles/roving tabindex/teclado intactos.
2. **`Button.tsx`** — confirmar variantes con radios nuevos; ajuste minimo si hace falta.
3. **`Modal.tsx`** — `bg-surface`, `border-border`, radio panel.
4. **`StatTile.tsx`** (nuevo) — `{ value, label, accent? }`, numero mono grande + label
   mono chico; `accent` pinta el valor en cian.
5. **`Badge.tsx`** (nuevo) — `{ variant: 'live'|'rec'|'active', children }`, pill con
   punto; live/rec rojo con `recpulse`, active verde con glow.
6. **`SectionLabel.tsx`** (nuevo) — `<div class="lbl">` con children.
7. **Eliminar `Switch.tsx`** (tras quitar su unico consumidor en Slice 2).

**Verificacion:** `npm run typecheck` + `build`.

---

## Slice 2 — Title bar + limpieza de tema

**Depende de:** Slice 1 (segmentado).

1. **`uiStore.ts`** — quitar `theme` / `toggleTheme` y la aplicacion de `.dark`.
   Mantener `activeView`/`setView`.
2. **`Header.tsx`** — title bar del spec: marca (`ReticleMark` + wordmark con "Lens" en
   cian), nav con el segmentado (Inferencia/Modelos), cluster derecho = modelo activo
   (nombre mono + badge de formato, solo lectura desde `activeModel` del workspaceStore).
   Quitar el `<ThemeToggle/>`.
3. **Eliminar** `ThemeToggle.tsx` y `Switch.tsx`. Buscar y limpiar imports muertos.

**Verificacion:** `npm run typecheck` + `build`; confirmar que no quedan referencias a
`theme`/`Switch`/`ThemeToggle`.

---

## Slice 3 — Inferencia: layout de 3 zonas

**Depende de:** Slices 1-2. Es el corazon del trabajo.

1. **`ModelRow.tsx`** (nuevo, en `features/inference/components`) — fila del spec:
   punto + nombre mono + badge de formato (de la extension). Activa = `bg-accent-soft`
   + borde cian + punto con glow; inactiva = `bg-control` + punto apagado. Recibe
   `entry`, `active`, `onSelect`.
2. **`ModelSelector` -> lista** — reusar `useModels`/`useSelectModel`/`activeModel`,
   renderizando `ModelRow`. (El selector deja el Header; el Header solo muestra activo.)
3. **`CameraSource` / `FileSource`** — re-skin de selects/labels (`bg-control`, `.lbl`).
4. **`ConfidenceSlider`** — label `.lbl`, valor mono cian (el thumb/pista vienen del css
   del Slice 0).
5. **`MetricsHUD`** — separar: el overlay sobre el feed se mantiene; agregar el **panel**
   de la derecha con grilla de `StatTile` (FPS/Inf/Total/P95->cian) leyendo `useMetrics`.
   (Puede ser un componente `MetricsPanel` nuevo que comparte el hook.)
6. **`LogPanel`** — items de error del spec (punto severidad + timestamp mono + mensaje
   sans; vacio = fila verde "Sin errores · pipeline estable").
7. **`Recorder`** — botones circulares rec/stop + `Badge` REC al grabar (barra de
   transporte).
8. **`VisionWorkspace`** — `bg-black` -> `bg-feed`; fondo de grilla sutil (cian 5%, 30px);
   `Badge` EN VIVO arriba-derecha en modo camara.
9. **`InferenceView`** — `grid-cols-[200px_1fr_230px]`: izq (Fuente + Modelo),
   centro (feed + transporte), der (Parametros + Metricas + Errores + boton Config
   avanzada). Metricas y errores **permanentes** (sin toggles).

**Verificacion:** `npm run typecheck` + `build`; revision visual en Electron del layout,
estados activos, badges, slider, grilla del feed.

---

## Slice 4 — Vista Modelos + wizard (solo estilos)

**Depende de:** Slices 1-2.

1. **`ModelsView`** — titulo "Modelos" como `h1` (Space Grotesk 700 25px); contenedores
   `bg-surface` + radios panel.
2. **`ModelCard`** — badge de formato mono; estado config con colores de sistema;
   seleccionada = borde cian + `bg-accent-soft`.
3. **`ModelDropzone`** — `bg-control`, borde punteado, acento cian en hover/drag.
4. **Wizard (`Step1`–`Step4`, `fields.tsx`)** — segmentado para tipo/runtime; inputs
   sobre `bg-control` + labels `.lbl`; pasos con `SectionLabel`. **Solo presentacion**
   (la estructura se reescribe en trabajo futuro — no reorganizar pasos/campos).

**Verificacion:** `npm run typecheck` + `build`; revision visual del wizard.

---

## Slice 5 — Verificacion final

1. `npm run typecheck` y `npm run build` limpios.
2. Checklist manual en Electron (`npm run dev` + `npm run start:dev`): fuentes offline,
   contraste, layout 3 zonas, activos en cian, badges/pulsos, slider, vista Modelos +
   wizard.
3. Actualizar `CLAUDE.md` (seccion frontend) y memoria si corresponde.

---

## Notas

- Trabajo guiado con la skill **interface-design** (craft-first) por slice.
- Riesgo principal: `MetricsHUD` cumple doble rol (overlay vs panel). Resolverlo
  extrayendo un `MetricsPanel` que comparte `useMetrics`, sin tocar el hook.
- Las fuentes embebidas en el spec son subset latin: validar que cubren los textos en
  espanol (acentos en UI, no en codigo). Si falta glyph, caer a system-ui via el stack.
