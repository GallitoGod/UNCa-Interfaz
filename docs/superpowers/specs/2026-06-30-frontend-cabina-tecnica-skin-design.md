# Re-skin del frontend a la piel "Cabina Tecnica" — diseno

**Fecha:** 2026-06-30
**Estado:** aprobado, listo para plan de implementacion
**Alcance:** solo aspecto visual del frontend React (`client/`). No cambia logica
(hooks, estado, inferencia, contratos con el backend).

## 1. Objetivo

Transformar el lenguaje visual del frontend React para que use la "plantilla de
estilos" Cabina Tecnica definida en `docs/UNCaLens_Spec_Visual.html`: paleta
casi-negro con un solo acento cian (`#34d6ff`), tipografias Space Grotesk +
JetBrains Mono, tokens de geometria (radios/bordes/espaciado), una libreria de
componentes con sus estados, y la anatomia de 3 zonas de la pantalla de Inferencia.

El spec describe **como debe verse**, no que hace la logica. La logica ya existe en
React; este trabajo cambia superficies, color, tipografia, componentes y layout.

## 2. Decisiones tomadas (brainstorming)

1. **Alcance:** re-skin **+** recomposicion de la pantalla de Inferencia al layout de
   3 zonas del spec (no solo swap de tokens).
2. **Tema:** **dark-only**. Se adopta la paleta del spec como unico tema y se elimina
   el toggle light/dark.
3. **Tipografia:** **fuentes empaquetadas localmente** (`.woff2` de Space Grotesk +
   JetBrains Mono en el bundle de Vite). Render fiel al spec y sigue funcionando
   offline en Electron.
4. **Componentes sin logica de respaldo:** solo se re-pinta **lo ya cableado**. Se
   **omiten** los elementos que el spec dibuja pero no tienen backend hoy: chips de
   clase poblados por detecciones y el slider de IoU/NMS. No se inventa logica nueva.
5. **Vistas:** se re-pintan **Inferencia + Modelos** (app coherente).
6. **Wizard de config:** su **estructura cambiara mas adelante**; en este trabajo solo
   recibe estilos (la recomposicion estructural es trabajo futuro fuera de este spec).

## 3. Enfoque elegido (B): design-system + recomposicion

Tres capas limpias, aprovechando que el frontend ya tiene tokens semanticos en
`index.css` y primitivos en `shared/ui`:

1. **Tokens** — reescribir `index.css` a la paleta/geometria/tipografia del spec.
2. **Primitivos** — ajustar los existentes y agregar los que el spec define y faltan.
3. **Vistas** — recomponer `InferenceView` a 3 zonas y re-pintar `ModelsView` + wizard.

## 4. Seccion 1 — Fundaciones (tokens + fuentes)

Reescribir `client/src/index.css` como **dark-only**. Mapeo token semantico -> valor:

| Token (utilidad)            | Valor spec               | Uso                          |
|-----------------------------|--------------------------|------------------------------|
| `--c-canvas` (`bg-canvas`)  | `#0a0d13`                | ventana                      |
| `--c-surface` (`bg-surface`)| `#0c1017`                | paneles laterales            |
| `--c-control` (`bg-control`)| `#11161f`                | controles / filas            |
| `--c-surface-raised`        | `#0e131b`                | tarjetas de propiedad (tile2)|
| `--c-feed` (nuevo)          | `#05080c`                | viewport del video           |
| `--c-border`                | `rgba(255,255,255,.06)`  | borde base                   |
| `--c-border-strong`         | `rgba(255,255,255,.10)`  | borde fuerte                 |
| `--c-accent`                | `#34d6ff`                | accion / activo / dato vivo  |
| `--c-accent-soft` (nuevo)   | `rgba(52,214,255,.12)`   | fondo de chip/fila activa    |
| `--c-accent-fg` (ink)       | `#04121a`                | texto sobre cian             |
| `--c-fg`                    | `#e7edf5`                | texto primario               |
| `--c-fg-muted`              | `#aab8cc`                | secundario                   |
| `--c-fg-subtle`             | `#8597ad`                | terciario                    |
| `--c-label` (nuevo)         | `#5b6b82`                | etiquetas mono mayuscula     |
| `--c-success`               | `#28c840`                | estado ok                    |
| `--c-warn`                  | `#febc2e`                | estado degradado             |
| `--c-danger`                | `#ff4d4f`                | fallo / REC                  |

**Geometria** (escala del spec):
- `--radius-sm: 6px` (chip)
- nuevo radio control `8px` (se puede exponer como `--radius` o usar arbitrary value)
- `--radius-md: 10px` (card)
- `--radius-lg: 13px` (panel)

**Fuentes:** agregar `.woff2` de `Space Grotesk` (pesos 400/500/600/700) y
`JetBrains Mono` (400/500/600/700) en `client/src/assets/fonts/`, declararlos con
`@font-face` + `font-display: swap`, y actualizar:
- `--font-sans: 'Space Grotesk', system-ui, sans-serif;`
- `--font-mono: 'JetBrains Mono', ui-monospace, monospace;`

**Utilidades de texto** del spec (en `@layer components` o equivalente):
- `.lbl` — mono 10px, `letter-spacing: 1.6px`, uppercase, color `--c-label`.
- `.swatch-meta` — mono 11px, color `--c-fg-subtle`.

**Eliminar** el bloque `:root` del tema claro y el `@custom-variant dark`. La app
queda en un solo tema; las utilidades `bg-*`/`text-*`/`border-*` siguen funcionando
porque leen las variables `--c-*` (ahora con un unico set de valores).

`@theme` se amplia con `--color-feed` y `--color-accent-soft` para exponer
`bg-feed` y `bg-accent-soft` / `border-accent-soft`.

## 5. Seccion 2 — Primitivos (`shared/ui`)

**Ajustar existentes:**
- **`Button`** — ya correcto: `primary` = `bg-accent text-accent-fg` (cian solido +
  ink). Verificar radios nuevos. Sin cambios estructurales.
- **`Tabs`** -> control **segmentado** del spec: pista `bg-control` con `padding:3px`,
  activo = `bg-accent text-accent-fg`, inactivo = `text-fg-subtle hover:text-fg`.
  Mantener todo el contrato a11y (roles tablist/tab, roving tabindex, flechas,
  Home/End). Mismo primitivo sirve a la nav del Header y a Camara/Archivo.
- **`ConfidenceSlider`** — re-estilar el `<input type=range>`: pista `#1b2330` 5px,
  fill cian, thumb blanco con glow via `::-webkit-slider-thumb` + `::-moz-range-thumb`
  (reglas en `index.css`), label con `.lbl`, valor a la derecha en mono cian.
- **`Modal`** — superficie `bg-surface`, borde `border`, radio panel (13px).

**Nuevos primitivos:**
- **`StatTile`** — numero grande JetBrains Mono + label mono chico; prop `accent?`
  para pintar el dato destacado en cian. Lo usa `MetricsHUD` / panel de metricas.
- **`Badge`** — pill redondeado con punto; variantes `live` / `rec` (rojo, punto que
  pulsa con `@keyframes recpulse`) y `active` (verde con glow).
- **`SectionLabel`** — wrapper trivial sobre `.lbl` para encabezados de seccion.

**Eliminar:** `Switch.tsx` y `ThemeToggle.tsx`.

**No se crean:** `ClassChips` ni slider de IoU (decision 4: solo lo cableado). La fila
de modelo (`ModelRow`) se arma en la Seccion 3 dentro de la feature porque depende de
datos del selector de modelos.

## 6. Seccion 3 — Inferencia: layout de 3 zonas

Reescribir `InferenceView.tsx` de 2 columnas a `grid-cols-[200px_1fr_230px]`, fondo
`bg-canvas`. El Header pasa a ser title bar (Seccion 5).

**Zona izquierda** (`bg-surface`, borde derecho, scroll):
- `SectionLabel` "Fuente" -> segmentado **Camara/Archivo** (Tabs) +
  `CameraSource`/`FileSource` re-skinneados (selects sobre `bg-control`).
- `SectionLabel` "Modelo" -> **lista de `ModelRow`** alimentada por `useModels` /
  `useSelectModel` / `activeModel` (logica ya cableada, solo re-presentada). Cada
  fila: punto + nombre (mono) + badge de formato (ONNX/TFLite/... por extension).
  Activa = `bg-accent-soft` + borde cian + punto con glow; inactiva = `bg-control` +
  punto apagado. **Reubica** la seleccion de modelo del dropdown del Header a esta
  lista. *Sin chips de clase.*

**Zona central** (columna flex):
- `VisionWorkspace` (feed, heroe): agregar fondo de **grilla sutil** del spec
  (`linear-gradient` 30px, cian 5%) y cambiar `bg-black` -> `bg-feed`. Overlay `Badge`
  **EN VIVO** (camara) arriba-derecha; `MetricsHUD` sigue como overlay opcional
  arriba-izq (independiente del panel de metricas de la derecha).
- **Barra de transporte** debajo (`bg-surface`, borde superior): controles de
  grabacion (`Recorder` -> botones circulares rec/stop del spec, `Badge` REC al
  grabar) y, en modo archivo, el scrubber si existe. **No** se agrega boton "play"
  (la camara no tiene play discreto cableado).

**Zona derecha** (`bg-surface`, borde izquierdo, scroll):
- `SectionLabel` "Parametros" -> `ConfidenceSlider` re-estilado. *(Sin slider IoU.)*
- `SectionLabel` "Metricas" -> grilla de `StatTile` (FPS, Inf, Total, P95->cian)
  leyendo `useMetrics`. Pasa de overlay-toggle a **panel permanente** (sigue haciendo
  polling solo mientras la vista esta montada).
- `SectionLabel` "Errores" -> `LogPanel` re-estilado como items de error (punto de
  severidad + timestamp mono + mensaje sans; vacio = fila verde "Sin errores ·
  pipeline estable"). Tambien **permanente**.
- Boton "Configuracion avanzada" -> abre `DrawSettingsModal` (sin cambios de logica).

**Cambio de UX (aprobado):** metricas y errores dejan de ser toggles y viven siempre
en el panel derecho, como el spec.

## 7. Seccion 4 — Vista Modelos + wizard

`ModelsView` y `ModelCard` ya consumen los tokens, asi que heredan la piel. Toques:
- **Titulo "Modelos"** como `h1` con la escala del spec (Space Grotesk 700 · 25px).
- **`ModelCard`** -> tarjeta del spec: badge de formato mono, punto de estado de
  config con colores de sistema (verde `● config` / apagado `○ sin config`),
  seleccionada = borde cian + `bg-accent-soft`.
- **`ModelDropzone`** -> superficie `bg-control`, borde punteado, acento cian en
  hover/drag.
- **Wizard (`Step1`–`Step4` + `fields.tsx`)**: elecciones de tipo
  (detection/classification/segmentation) y de runtime (CPU/GPU, FP32/FP16) con el
  **segmentado**; inputs/labels sobre `bg-control` + `.lbl`; pasos encabezados con
  `SectionLabel`. **Solo presentacion** — la estructura del wizard se reescribira en un
  trabajo futuro, asi que se aplican estilos sin reorganizar pasos ni campos.

## 8. Seccion 5 — Title bar + limpieza

- **Header -> title bar del spec**: marca (`ReticleMark` + wordmark "UNCa**Lens**" con
  "Lens" en cian) · nav **segmentado** Inferencia/Modelos · cluster derecho = **modelo
  activo** (nombre mono + badge de formato, solo lectura desde `activeModel`). Se quita
  el `ThemeToggle`.
- **Eliminar** `ThemeToggle.tsx` y `Switch.tsx`; en `uiStore` sacar el estado
  `theme`/`toggleTheme` y la aplicacion de la clase `.dark` (queda solo la navegacion
  de vistas). Verificar que ningun otro consumidor dependa de esos.
- **Fuentes** (cubierto en Seccion 1): `client/src/assets/fonts/*.woff2`.
- **Fuera de alcance:** el frontend viejo `src/render/` + `static/` (ya marcado
  muerto/removible en CLAUDE.md) no se toca.

## 9. Componentes y archivos afectados (resumen)

- `client/src/index.css` — tokens, fuentes, utilidades, slider/keyframes.
- `client/src/assets/fonts/` — `.woff2` nuevos.
- `client/src/shared/ui/`: `Button` (verif), `Tabs` (segmentado), `Modal` (skin),
  `StatTile` (nuevo), `Badge` (nuevo), `SectionLabel` (nuevo); **eliminar** `Switch`.
- `client/src/app/components/`: `Header` (title bar), **eliminar** `ThemeToggle`.
- `client/src/app/store/uiStore.ts` — quitar tema.
- `client/src/features/inference/`: `InferenceView` (3 zonas), `ModelRow` (nuevo),
  `CameraSource`/`FileSource` (skin), `ConfidenceSlider` (skin), `MetricsHUD` (+ panel
  de StatTiles), `LogPanel` (items de error), `Recorder` (transporte), `ModelSelector`
  (pasa a lista; el Header solo muestra activo).
- `client/src/features/vision-workspace/components/VisionWorkspace.tsx` — feed/grilla.
- `client/src/features/models/`: `ModelsView`, `ModelCard`, `ModelDropzone`, wizard
  (`Step1`–`Step4`, `fields.tsx`) — solo skin.

## 10. Verificacion

- No hay test runner en `client/`. Verificar con `npm run typecheck` (tsc --noEmit) y
  `npm run build` (vite build) — ambos deben pasar.
- Verificacion visual real corriendo Electron es manual (igual que la deuda #18 del
  CLAUDE.md): contraste, fuentes cargando offline, layout de 3 zonas, estados activos
  en cian, badges/pulsos, slider.

## 11. Fuera de alcance / no objetivos

- No se cambia ninguna logica, hook, store (salvo quitar el tema), contrato WS/HTTP ni
  el pipeline de inferencia.
- No se implementan chips de clase ni control de IoU (sin backend).
- No se reorganiza la **estructura** del wizard (solo estilos; reescritura futura).
- No se toca el frontend viejo `src/render/` + `static/`.
- No se agrega tema claro.
