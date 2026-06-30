# Simplificacion del wizard de configuracion de modelos — Diseño

**Fecha:** 2026-06-30
**Estado:** aprobado (pendiente revision del usuario)
**Alcance:** poda de parametros del `ConfigWizard` (`client/src/features/models/components/ConfigWizard/`).
El reskin visual "Cabina Tecnica" de la seccion Modelos queda como fase siguiente, separada de este spec.

---

## 1. Objetivo

El wizard de configuracion de modelos expone hoy ~12 controles por paso, muchos de los
cuales son **derivables de otro campo**, **codigo muerto**, o **parametros que casi nunca
se tocan**. Eso vuelve el wizard intimidante y propenso a configuraciones contradictorias.

Meta: reducir la superficie editable a lo que el usuario realmente decide, derivando o
preconfigurando el resto, sin perder potencia (los casos raros siguen accesibles bajo
"Avanzado"). Solo se configura **Deteccion**; Clasificacion y Segmentacion quedan visibles
pero deshabilitadas hasta que el backend implemente sus pipelines (hoy levantan
`TaskNotImplemented` -> 501).

## 2. Principio rector

**Campo maestro -> campos derivados.** Igual que el slider de confianza ya se lee en vivo,
varios campos del wizard son funciones de otro:

- `channels` es funcion de `color_order` (RGB/BGR -> 3, GRAY -> 1).
- `tensor_structure` + `out_coords_space` son constantes conocidas por `pack_format`.
- providers (ONNX) / delegates (TFLite) son funcion de `device`.

El wizard expone el campo maestro y **deriva** los subordinados. El usuario no ve ni edita
los derivados salvo que abra "Avanzado".

## 3. Invariante critica

El wizard **siempre emite un `ModelConfig` completo y valido** contra el schema estricto
(`extra="forbid"`). Los campos ocultos reciben valores derivados/preset; **nunca se omiten**.
Esto es lo que evita romper el `POST /configs/{name}`, que valida contra `ModelConfig`
antes de escribir.

## 4. Cambios por paso

### Paso 1 — Tipo (`Step1Type.tsx`)

- Se mantienen las **3 tarjetas**.
- Solo **Deteccion** es clickeable.
- **Clasificacion** y **Segmentacion** se renderizan deshabilitadas (mas oscuras,
  `cursor: not-allowed`, sin `onClick`), con una pista visual de "proximamente".
- Motivo: sus pipelines no estan implementados en el backend (501). Configurarlos hoy
  produce JSONs que no pueden correr.

### Paso 2 — Input (`Step2Input.tsx`)

| Control actual | Cambio |
|---|---|
| `quantized` (checkbox) | **Eliminado.** Es codigo muerto: no se consume en ningun loader/adapter/transformer (solo aparece en `config_schema.py:19` y 2 tests). `dtype` (int8/uint8) ya es el unico knob real de cuantizacion. |
| `channels` (campo numerico) | **Eliminado de la UI.** Se deriva de `color_order` (RGB/BGR -> 3, GRAY -> 1) al armar el JSON. `input_adapter.py:30-35` ya trata `channels` como subordinado a `color_order`, asi que esto solo elimina la doble fuente de verdad. El schema sigue requiriendo `channels`: lo escribe el wizard. |
| `scale` + `normalize` + 6 cajas mean/std | **Reemplazado por un preset** (selector unico): `Ninguna` / `Escalar [0,1]` / `ImageNet` / `Personalizado`. Las 6 cajas mean/std solo aparecen en `Personalizado`. Mapeo a campos backend abajo. |
| `layout` + `dtype` | **Movidos a seccion "Avanzado"** (colapsada por defecto). Son reales y necesarios pero el caso comun no los toca. |
| `letterbox` + `preserve_aspect_ratio` (2 checkboxes que interactuan) | **Reemplazados por un selector geometrico unico**: `Resize directo (deforma)` / `Letterbox (preserva + padding)`. Elimina la combinacion contradictoria (hoy `use_letterbox = letterbox AND preserve_aspect_ratio`, `input_transformer.py:113`). |
| `auto_pad_color` (3 cajas numericas) | **Color picker** (solo visible cuando el modo geometrico es Letterbox). |

**Sin preview en vivo.** Se evaluo mostrar una imagen de ejemplo adaptada; se descarto:
para normalizacion el resultado no es interpretable por un humano (valores negativos);
para letterbox el valor no justifica el costo en esta fase.

**Mapeo preset de normalizacion -> campos backend:**

| Preset | `scale` | `normalize` | `mean` | `std` |
|---|---|---|---|---|
| Ninguna | `false` | `false` | `[0,0,0]` | `[1,1,1]` |
| Escalar [0,1] | `true` | `false` | `[0,0,0]` | `[1,1,1]` |
| ImageNet | `true` | `true` | `[0.485,0.456,0.406]` | `[0.229,0.224,0.225]` |
| Personalizado | `true` | `true` | (cajas) | (cajas) |

El preset activo se infiere al cargar una config existente comparando los valores contra
las tablas (si no coincide con ningun preset conocido -> `Personalizado`).

### Paso 3 — Output / Deteccion (`Step3Output.tsx`, rama `DetectionStep`)

- `pack_format` pasa a ser **preset maestro**. Al elegirlo, autocompleta:
  - `tensor_structure` (box_format, coordinates, confidence_index, class_index) con las
    constantes correctas del unpacker de ese formato.
  - `runtimeShapes.out_coords_space` (ej: `anchor_deltas` -> `tensor_pixels`,
    `tflite_detpost` -> `normalized_0_1`).
  - `anchor_config` (ya existente) para `anchor_deltas`.
- La **"Estructura por deteccion"** (los indices/coords) deja de estar visible por defecto.
  Pasa a un acordeon **"Avanzado"** como override editable, disponible en cualquier formato
  (para exports raros que difieran del unpacker estandar). Para `raw` conviene mostrarla
  expandida por defecto, porque ese formato es por definicion "decime vos donde esta cada cosa".
- **Filtrado y NMS:**
  - Se quita el checkbox `apply_conf_filter`.
  - Se quita tambien el campo `confidence_threshold` del wizard: el umbral se ajusta en
    vivo con el slider de inferencia, asi que el valor del JSON es solo el inicial (queda
    el default del template). `apply_conf_filter` conserva el valor del template/config.
  - `nms_per_class` -> "Avanzado".
  - `top_k`, `apply_nms` + `nms_threshold` se mantienen visibles.

**Constantes por `pack_format` (a fijar en implementacion):** los indices exactos de
`tensor_structure` por formato se obtienen **leyendo cada unpacker** (`unpackers/*.py`) y
el flujo de `_NEEDS_ADAPTER` en `detection.py:20`. Esto es critico: un indice equivocado
reordena las cajas en silencio. El plan de implementacion debe pinear cada constante contra
el codigo del unpacker, no inventarla.

### Paso 4 — Runtime (`Step4Runtime.tsx`)

- Vista tipica: **Backend + Dispositivo**.
- Providers (ONNX: CUDA/CPU) y delegate (TFLite: gpu) se **derivan de `device`**
  (`gpu` -> CUDA/gpu-delegate primero; `cpu` -> CPU). El wizard escribe los arrays.
- `warmup.runs`, threads intra/inter (ONNX) / num_threads (TFLite), y un **override manual
  de providers/delegates** -> seccion "Avanzado" colapsada.

## 5. Componentes nuevos / tocados

- **Frontend (`client/`):**
  - `Step1Type.tsx`: tarjetas CLS/SEG deshabilitadas.
  - `Step2Input.tsx`: preset de normalizacion, selector geometrico, color picker,
    derivacion de channels, seccion Avanzado (layout/dtype).
  - `Step3Output.tsx`: pack_format como preset, Avanzado con tensor_structure, ajuste de
    Filtrado/NMS.
  - `Step4Runtime.tsx`: derivacion de providers/delegates desde device, seccion Avanzado.
  - `fields.tsx`: posibles primitivos nuevos — `ColorField` (color picker) y un contenedor
    colapsable `AdvancedSection` reutilizable.
  - `types.ts`: quitar `quantized` de `InputTensor`.
  - Logica de derivacion/preset: helpers puros (mapeo preset<->campos, device<->providers,
    pack_format<->tensor_structure) idealmente en un modulo aparte testeable de cabeza, no
    inline en los componentes.
- **Backend (cambio minimo):**
  - `config_schema.py:19`: quitar `quantized: bool = False` de `InputTensor`.
  - `tests/test_load_model_config.py:31,96`: quitar `"quantized": False` de los fixtures.
  - **Sanear los 5 configs** que declaran `quantized` (si no, el schema estricto los
    rechaza al cargar): `configs/yolov7-tiny.json:30`, `configs/efficientdet-lite0.json:30`,
    `configs/efficientdet-lite2.json:30`, `configs/plantillas/estructura.json:30`,
    `configs/plantillas/estructuraEtapa5.json:30`. Quitar la linea `"quantized": false`.

## 6. Lo que NO cambia

- El contrato del pipeline de inferencia (envelope WS, formato `[x1,y1,x2,y2,conf,cls]`).
- Los endpoints (`POST /configs/{name}`, `GET /config/template/{model_type}`, etc.).
- El schema mas alla de quitar `quantized`.
- Los configs existentes en `configs/` siguen siendo validos (cargarlos en el wizard debe
  reconstruir el preset/modo correcto por inferencia de valores).

## 7. Riesgos

1. **Indices por pack_format mal fijados** -> cajas rotas en silencio. Mitigacion: pinear
   contra el codigo del unpacker + probar cada formato con un modelo real.
2. **Round-trip de configs existentes**: cargar una config previa debe mapear sus valores
   crudos de vuelta al preset/modo correcto. Si los valores no matchean ningun preset,
   caer a `Personalizado` / Avanzado expandido (sin perder datos).
3. **Quitar `quantized` del schema** rompe cualquier JSON que aun lo declare. Confirmado:
   los 5 configs lo declaran (ver seccion 5). Sanearlos en el mismo cambio es obligatorio,
   no opcional.

## 8. Verificacion

- `npm run typecheck` + `npm run build` en `client/`.
- `pytest` (el cambio de schema no debe romper tests tras sanear fixtures).
- Round-trip manual: cargar cada config existente de `configs/` en el wizard, confirmar que
  los presets/modos se infieren bien y que re-guardar produce un JSON equivalente.
- Verificacion visual en Electron (no automatizable): los 3 pasos, Avanzado colapsa/expande,
  CLS/SEG deshabilitados.
