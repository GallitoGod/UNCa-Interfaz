# Plan de implementacion — Simplificacion del wizard de Modelos

**Spec:** `docs/superpowers/specs/2026-06-30-wizard-modelos-simplificacion-design.md`
**Fecha:** 2026-06-30
**Regla transversal:** el wizard SIEMPRE emite un `ModelConfig` completo y valido contra el
schema estricto. Comentarios/docstrings en espanol sin tildes (convencion del repo).
Verificacion tras cada slice: `npm run typecheck` + `npm run build` en `client/`; `pytest`
para el backend. Se puede commitear por slice.

Orden por dependencia. El slice 1 (constantes/helpers puros) no toca UI y desbloquea el resto.

---

## Slice 0 — Backend: eliminar `quantized` (codigo muerto)

**Depende de:** nada.

1. `src/api/func/reader_pipeline/config_schema.py:19` — quitar `quantized: bool = False`
   de `InputTensor`.
2. `src/api/func/reader_pipeline/tests/test_load_model_config.py:31,96` — quitar
   `"quantized": False` de ambos fixtures.
3. **Sanear los 5 configs** (el schema estricto rechaza el campo): quitar la linea
   `"quantized": false` de:
   - `configs/yolov7-tiny.json`
   - `configs/efficientdet-lite0.json`
   - `configs/efficientdet-lite2.json`
   - `configs/plantillas/estructura.json`
   - `configs/plantillas/estructuraEtapa5.json`

**Verificacion:** `pytest` verde. `GET /configs/{name}` carga los 3 modelos sin 422.

---

## Slice 1 — Helpers puros de preset/derivacion (`lib/wizardPresets.ts`)

**Depende de:** nada (no toca UI). **Bloquea:** slices 2-4.

Modulo nuevo `client/src/features/models/lib/wizardPresets.ts`, funciones puras y
testeables de cabeza (sin store, sin React):

1. **Normalizacion** (preset <-> campos input):
   - `type NormPreset = 'none' | 'scale01' | 'imagenet' | 'custom'`
   - `NORM_PRESETS`: tabla preset -> `{scale, normalize, mean, std}` (ver spec §4 paso 2).
     ImageNet: mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
   - `inferNormPreset(input) -> NormPreset`: compara `scale/normalize/mean/std` contra la
     tabla; si no matchea ninguno -> `'custom'`.
   - `applyNormPreset(preset)`: devuelve los 4 campos a setear (el componente los aplica
     con setField; `custom` no toca mean/std, solo asegura `scale=true, normalize=true`).

2. **pack_format -> estructura de salida** (constantes fijadas contra unpackers + configs
   reales, ver spec §4 paso 3):
   - `PACK_PRESETS`: por formato `{tensor_structure, out_coords_space}`:
     - `yolo_flat`: box `cxcywh` `{cx:0,cy:1,w:2,h:3}`, conf 4, cls 5; `tensor_pixels`.
     - `tflite_detpost`: box `yxyx` `{y1:0,x1:1,y2:2,x2:3}`, conf 4, cls 5; `normalized_0_1`.
     - `anchor_deltas`: box `yxyx` `{y1:0,x1:1,y2:2,x2:3}`, conf 4, cls 5; `tensor_pixels`.
     - `boxes_scores`: box `xyxy` `{x1:0,y1:1,x2:2,y2:3}`, conf 4, cls 5; `normalized_0_1`
       (no usa adapter; los indices son solo para validar el schema).
     - `raw`: NO se autocompleta (queda lo que haya / lo edita el usuario).
   - `applyPackPreset(format)`: devuelve `{tensor_structure?, out_coords_space, num_classes}`
     a setear; preserva `num_classes` actual (es del modelo, no del formato).
   - **Nota de exactitud:** estas constantes salen de `output_adapter.py` (orden que lee el
     box converter) + el orden de columnas que stackea cada unpacker + los configs
     `efficientdet-lite0/2.json` (anchor_deltas, verificados). No inventar.

3. **device -> ejecucion** (ONNX providers / TFLite delegates):
   - `providersForDevice(device) -> string[]`: `gpu` -> `['CUDAExecutionProvider',
     'CPUExecutionProvider']`; `cpu` -> `['CPUExecutionProvider']`.
   - `delegatesForDevice(device) -> string[]`: `gpu` -> `['gpu']`; `cpu` -> `[]`.

**Verificacion:** `npm run typecheck`. (No hay runner de tests JS; las funciones son
triviales y se validan en el round-trip manual del slice 5.)

---

## Slice 2 — Primitivos de campo nuevos (`fields.tsx`)

**Depende de:** nada.

1. **`ColorField`** — color picker que mapea `number[] [r,g,b]` <-> hex. `<input type=color>`
   estilado + (opcional) el valor hex/RGB en mono al lado. Maneja `null` -> default `[114,114,114]`.
2. **`AdvancedSection`** — contenedor colapsable (`<details>`/`<summary>` o estado local) con
   titulo "Avanzado", cerrado por defecto. Reusable en pasos 2/3/4.
3. **`SegmentedField`** (si conviene) — selector segmentado para reemplazar selects binarios
   (preset geometrico, etc.). Si es overkill, usar `SelectField` existente. Decidir al
   implementar el paso 2.

**Verificacion:** `npm run typecheck` + `build`.

---

## Slice 3 — Paso 1: deshabilitar CLS/SEG (`Step1Type.tsx`)

**Depende de:** nada.

1. Mantener las 3 tarjetas. Solo `detection` clickeable.
2. `classification`/`segmentation`: render deshabilitado (mas oscuro, `opacity` reducida,
   `cursor-not-allowed`, sin `onClick`), con etiqueta "Proximamente". `aria-disabled`.

**Verificacion:** `npm run typecheck` + `build`.

---

## Slice 4 — Paso 2: Input simplificado (`Step2Input.tsx`)

**Depende de:** slices 1-2.

1. **Dimensiones:** Ancho, Alto, Orden de color. **Quitar el campo Canales.**
   Derivar `channels` de `color_order` (GRAY -> 1, RGB/BGR -> 3) y escribirlo con setField
   cuando cambia `color_order` (mantiene el JSON valido).
2. **Normalizacion:** un `SelectField`/segmentado con `NormPreset`. Preset actual via
   `inferNormPreset(inp)` + estado local para respetar la eleccion `custom` del usuario.
   Al cambiar: aplicar `applyNormPreset`. Las 6 cajas mean/std solo en `custom`.
   **Quitar el checkbox `quantized`** (ya no existe en el tipo).
3. **Avanzado:** `layout` + `dtype` dentro de `AdvancedSection`.
4. **Letterbox (solo deteccion):** reemplazar los 2 checkboxes por un selector geometrico:
   `resize` (`letterbox=false`) / `letterbox` (`letterbox=true, preserve_aspect_ratio=true`).
   Valor derivado de `inp.letterbox && inp.preserve_aspect_ratio`. `auto_pad_color` ->
   `ColorField`, visible solo en modo letterbox.

**Verificacion:** `npm run typecheck` + `build`.

---

## Slice 5 — Paso 3: Output deteccion (`Step3Output.tsx`)

**Depende de:** slices 1-2.

1. **`pack_format` como preset maestro:** en `onPackFormat`, ademas de setear el formato,
   aplicar `applyPackPreset` (tensor_structure + out_coords_space) salvo para `raw`.
   Mantener el autocompletado de `anchor_config` ya existente para `anchor_deltas`.
2. **`out_coords_space`:** se mantiene visible como select (decision real), pre-sembrado por
   el preset. Para `anchor_deltas` queda forzado a `tensor_pixels` (deshabilitado).
3. **"Estructura por deteccion" -> `AdvancedSection`** (override): box_format, coordinates,
   confidence_index, class_index. Para `pack_format === 'raw'` mostrarla expandida por
   defecto (es manual por definicion). `num_classes` se mantiene visible (es del modelo).
4. **Filtrado y NMS:** quitar el checkbox `apply_conf_filter`; setearlo derivado
   (`confidence_threshold > 0`). `nms_per_class` -> `AdvancedSection`. Mantener visibles
   `confidence_threshold`, `top_k`, `apply_nms`, `nms_threshold` (este ultimo si `apply_nms`).
5. **CLS/SEG:** `ClassificationStep`/`SegmentationStep` quedan intactos (inalcanzables porque
   el paso 1 los deshabilita; fuera de alcance de esta poda).

**Verificacion:** `npm run typecheck` + `build`.

---

## Slice 6 — Paso 4: Runtime simplificado (`Step4Runtime.tsx`)

**Depende de:** slice 1.

1. **Vista tipica:** `backend` + `device`. Al cambiar `device`, derivar y escribir
   `onnx.providers` (`providersForDevice`) y `tflite.delegates` (`delegatesForDevice`).
2. **Avanzado (`AdvancedSection`):** `warmup.runs`, threads (intra/inter para ONNX,
   num_threads para TFLite), y el override manual de providers/delegates (los checkboxes
   actuales). El bloque por-backend (ONNX/TFLite) vive dentro de Avanzado.

**Verificacion:** `npm run typecheck` + `build`.

---

## Slice 7 — Verificacion final

1. `pytest` verde; `npm run typecheck` + `npm run build` limpios.
2. **Round-trip manual en Electron** (`npm run dev` + `npm run start:dev`): abrir el wizard
   sobre cada config existente (`yolov7-tiny` raw, `efficientdet-lite0/2` anchor_deltas) y
   confirmar:
   - presets/modos se infieren correctamente (normalizacion, geometrico, pack_format);
   - re-guardar produce un JSON equivalente al original (sin `quantized`, con channels y
     tensor_structure correctos);
   - el backend acepta el guardado (sin 422).
3. Crear un modelo nuevo de cada `pack_format` y verificar que el preset deja una config
   valida.
4. Confirmar CLS/SEG deshabilitados en paso 1; Avanzado colapsa/expande en pasos 2-4.
5. Actualizar `CLAUDE.md` (mapa frontend + estado) y memoria si corresponde.

---

## Riesgos y mitigacion

- **Indices por pack_format:** fijados contra unpackers + configs reales (slice 1). El
  round-trip del slice 7 sobre los 3 modelos reales es la red de seguridad.
- **Round-trip de configs viejos:** la inferencia de presets debe caer a `custom`/Avanzado
  si los valores no matchean, sin perder datos. Verificar en slice 7.
- **Derivaciones que pisan ediciones:** `channels` (de color_order) y providers (de device)
  se sobrescriben al cambiar el maestro; el override manual de providers vive en Avanzado
  para los casos raros.
