# App Shell (chrome común)

El "marco" de la aplicación: arranque del renderer, navegación entre vistas, tabs de
fuente, tema oscuro y la configuración del cliente backend. No contiene lógica de
inferencia ni de modelos; solo orquesta y monta el resto.

Archivos as-is: `static/index.html`, `src/render/scripts.js` (la parte de shell),
`src/render/modules/constants.js`.

> **✅ Estado (2026-06-26): implementado.** Construido en `client/src/app/` + `shared/`.
> Dos decisiones que estaban "abiertas/opcionales" quedaron **resueltas**:
> - **Pausa/reanudación del stream al navegar** (antes "mejora opcional"): al ir de Inferencia
>   a Modelos, `InferenceView` **no se desmonta** (queda oculto con `hidden`) y la sesión se
>   **pausa** (loop + `<video>` + captura de cámara con `track.enabled=false`) sin cerrar el
>   WS ni soltar el permiso; al volver, **reanuda**. Ver `videoStream.pause/resume` +
>   `useVisionSession` (en `feature-inferencia.md`). De paso arregló un bug: el cleanup viejo
>   revocaba el `objectURL` del file-video al navegar.
> - **Tema persistido en `localStorage`** (antes "mejora sugerida"): hecho en `uiStore`.

---

## Bootstrap

- **Responsabilidad:** Punto de entrada del renderer. Espera `DOMContentLoaded`, cachea
  referencias a los nodos del DOM y conecta todos los listeners de la UI (shell +
  controles de inferencia). Dispara `getModels()` y `initCameras()` al arrancar.
- **Entradas:** evento `DOMContentLoaded`; el DOM definido en `static/index.html`.
- **Salidas:** efectos de wiring (todos los `addEventListener`); arranque del flujo de
  modelos y de cámara.
- **Dependencias:** importa casi todos los módulos del renderer (`cameraSwitcher`,
  `record`, `modelLoader`, `selectModel`, `fileHandler`, `constants`, `modelsManager`,
  `overlay`). Es el único módulo que conoce a todos.
- **Reglas de negocio:**
  - Inicializa `drawSettings.bboxColor/labelColor` desde los `<input type=color>` antes
    de cualquier dibujo.
  - `getModels()` auto-selecciona el primer modelo para que el backend tenga uno cargado
    al arrancar (ver ModelSelector).
  - La vista "Modelos" se inicializa **perezosamente** la primera vez que se navega a
    ella (`modelsReady` flag), no al cargar la app.
- **Mapeo al destino React:** `src/app/App.tsx` + `src/app/providers/`. Desaparece como
  función monolítica: el wiring se reparte en componentes y hooks. Los `let intervalId`
  y referencias manuales al DOM se eliminan (React + Query los reemplazan). La
  inicialización perezosa de la vista Modelos la da el router (code-splitting por ruta).

## ViewRouter

- **Responsabilidad:** Alterna entre las dos vistas de primer nivel: **Inferencia** y
  **Modelos**. Es navegación local (no hay URL/rutas reales en la app actual).
- **Entradas:** clicks en `.nav-btn[data-view]` (header).
- **Salidas:** toggle de la clase `active` sobre `.nav-btn` y `.view`; en la primera
  entrada a "models" llama `initModelsManager()`.
- **Dependencias:** `modelsManager.initModelsManager`.
- **Reglas de negocio:** una sola vista activa a la vez; "models" se inicializa una única
  vez (flag `modelsReady`). No se desmonta la vista de inferencia al cambiar (el stream
  de cámara sigue vivo por debajo).
- **Mapeo al destino React:** `src/app/router.tsx`. Candidato a React Router (o estado en
  un store de UI si se prefiere navegación sin URL). La inicialización perezosa se vuelve
  lazy-loading de la feature `models`. **Decisión a tomar:** si al salir de Inferencia se
  pausa o no el WebSocket (hoy sigue corriendo).

## SourceTabs

- **Responsabilidad:** Dentro de la vista Inferencia, alterna el panel de control entre
  **Cámara** y **Cargar Video**.
- **Entradas:** clicks en `.tab-button[data-tab]`.
- **Salidas:** toggle de clase `active` sobre `.tab-button` y `.tab-content`.
- **Dependencias:** ninguna (puro DOM); los paneles que muestra pertenecen a
  feature-inferencia (CameraSource y FileSource).
- **Reglas de negocio:** solo presentación; cambiar de tab **no** detiene la fuente
  activa por sí mismo (eso lo hacen CameraSource/FileSource al iniciar una nueva fuente).
- **Mapeo al destino React:** `shared/ui/Tabs` + composición en
  `features/inference/components`. Estado de tab activa local al componente.

## ThemeToggle

- **Responsabilidad:** Activa/desactiva el modo oscuro.
- **Entradas:** `change` del checkbox `#dark-mode-toggle`.
- **Salidas:** toggle de la clase `dark` en `<html>`.
- **Dependencias:** CSS (`styles.css` usa la clase `dark`).
- **Reglas de negocio:** la app arranca en modo oscuro (`<html class="dark">` y checkbox
  `checked` por defecto). No persiste la preferencia (se reinicia en cada arranque).
- **Mapeo al destino React:** `shared/ui/ThemeToggle` con Tailwind (`darkMode: 'class'`).
  Mejora sugerida: persistir en `localStorage`. Estado en un store de UI liviano o
  contexto.

## BackendClient (constants)

- **Responsabilidad:** Única fuente de las URLs del backend (HTTP y WS). Puerto `8000`
  hardcodeado.
- **Entradas:** ninguna (constantes de módulo).
- **Salidas:** exporta `loadModelUrl`, `selectModelUrl`, `confidenceUrl`,
  `inferenceLogsUrl`, `metricsUrl`, `streamUrl`.
- **Dependencias:** ninguna.
- **Reglas de negocio:** host `127.0.0.1` y puerto `8000` fijos; backend local. No hay
  configuración de entorno.
- **Mapeo al destino React:** `shared/api/axios.ts` (instancia Axios con `baseURL`) +
  `shared/api/ws.ts` para la URL del WebSocket. El puerto/host pasan a variables de
  entorno de Vite (`import.meta.env`). Todas las llamadas `fetch` directas del código
  actual se reemplazan por esta instancia Axios + hooks de TanStack Query.

---

## Diseño to-be (React)

Esta sección documenta el destino. Profundidad escalada a complejidad: medio para
Bootstrap/Router/BackendClient, liviano para Tabs/Theme.

### Estructura

```
src/
├── app/
│   ├── App.tsx                # monta providers + router
│   ├── router.tsx            # ViewRouter (Inferencia | Modelos)
│   └── providers/
│       └── AppProviders.tsx  # QueryClientProvider + (theme)
└── shared/
    ├── api/
    │   ├── axios.ts          # instancia Axios (baseURL desde env)
    │   └── ws.ts            # URL del WebSocket desde env
    └── ui/
        ├── Tabs.tsx
        ├── ThemeToggle.tsx
        └── Modal.tsx
```

### Bootstrap → `app/App.tsx` + `providers/` *(medio)*

Reemplaza el monolito `DOMContentLoaded` de `scripts.js`. El wiring manual (cachear nodos,
`addEventListener`) desaparece: cada feature monta sus propios componentes.

```tsx
// app/providers/AppProviders.tsx
const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 5_000, retry: 1 } },
});

export function AppProviders({ children }: { children: React.ReactNode }) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

// app/App.tsx
export function App() {
  return (
    <AppProviders>
      <Header />        {/* logo + nav + ModelSelector + ThemeToggle */}
      <ViewRouter />
    </AppProviders>
  );
}
```

Regla heredada: el ModelSelector dispara la auto-selección del primer modelo al resolver
la query (ver `feature-inferencia.md`), no en el bootstrap.

### ViewRouter → `app/router.tsx` *(medio)*

Dos vistas de primer nivel. La feature Modelos se **carga perezosamente** (code-splitting)
en vez del flag `modelsReady` actual.

```tsx
const InferenceView = lazy(() => import('@/features/inference/InferenceView'));
const ModelsView = lazy(() => import('@/features/models/ModelsView'));

export function ViewRouter() {
  const view = useUiStore((s) => s.activeView); // 'inference' | 'models'
  return (
    <Suspense fallback={<Spinner />}>
      {view === 'inference' ? <InferenceView /> : <ModelsView />}
    </Suspense>
  );
}
```

**Decisión tomada:** navegación por estado en un `uiStore` (Zustand) liviano, sin URLs
(la app no las usa hoy). Si más adelante se quieren rutas reales, se cambia por React
Router sin tocar las features.

**Implementado (cambio respecto al plan):** `InferenceView` queda **montado siempre** y sólo
se oculta (`hidden`) cuando la vista es Modelos; `ModelsView` se monta/desmonta normal. Así se
preservan los refs `<video>/<canvas>/<overlay>`, la `MediaStream` y el WebSocket. Al navegar,
la sesión se **pausa** y al volver **reanuda** (`videoStream.pause/resume` disparados por un
effect sobre `uiStore.activeView` en `useVisionSession`). App-shell sólo cambia la vista; la
reacción vive en la feature de inferencia (el shell no conoce el WS).

```tsx
// router.tsx (real)
<Suspense fallback={<ViewFallback />}>
  <div className={cn('h-full', view !== 'inference' && 'hidden')}>
    <InferenceView />          {/* nunca se desmonta */}
  </div>
  {view === 'models' && <ModelsView />}
</Suspense>
```

### BackendClient → `shared/api/axios.ts` + `ws.ts` *(medio)*

Mata el `constants.js` con puerto hardcodeado.

```ts
// shared/api/axios.ts
const baseURL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000';
export const api = axios.create({ baseURL });

// shared/api/ws.ts
const wsBase = import.meta.env.VITE_WS_URL ?? 'ws://127.0.0.1:8000';
export const STREAM_URL = `${wsBase}/video_stream`;
```

Todos los `fetch` directos del código actual se reemplazan por `api.*` dentro de los hooks
de TanStack Query de cada feature. Ningún componente llama `fetch`/`axios` directo.

### SourceTabs → `shared/ui/Tabs` *(liviano)*

Componente de presentación controlado: `<Tabs value activeKey onChange>`. Estado de la tab
activa local al `InferenceView`. Sin lógica de negocio (cambiar de tab no detiene la
fuente; eso lo hacen CameraSource/FileSource).

### ThemeToggle → `shared/ui/ThemeToggle` *(liviano)*

`darkMode: 'class'` en Tailwind; togglea `dark` en `<html>`. **Mejora respecto al as-is:**
persistir en `localStorage` (hoy se reinicia en cada arranque). Estado en `uiStore` o un
hook `useTheme`.
