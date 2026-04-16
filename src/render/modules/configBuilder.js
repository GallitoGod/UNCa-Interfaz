const fs   = require('fs');
const path = require('path');

// ── Estado del builder ──────────────────────────────────────────────────────
let _state     = null;
let _builderEl = null;
let _cfgsDir   = null;
let _onSave    = null;

// ── Defaults por tipo ───────────────────────────────────────────────────────
const OUTPUT_DEFAULTS = {
  detection: {
    apply_conf_filter: true,
    confidence_threshold: 0.5,
    apply_nms: false,
    top_k: 0,
    nms_per_class: false,
    nms_threshold: 0.45,
    tensor_structure: {
      box_format: 'xyxy',
      coordinates: { x1: 1, y1: 2, x2: 3, y2: 4 },
      confidence_index: 6,
      class_index: 5,
      num_classes: 80
    },
    pack_format: 'raw',
    out_coords_space: 'tensor_pixels'
  },
  classification: {
    apply_softmax: true,
    apply_sigmoid: false,
    top_k: 1,
    confidence_threshold: 0.5,
    label_map: null,
    tensor_structure: {
      num_classes: 1000,
      output_format: 'logits',
      multi_label: false
    },
    pack_format: 'softmax_out'
  },
  segmentation: {
    confidence_threshold: 0.5,
    label_map: null,
    tensor_structure: {
      num_classes: 21,
      output_format: 'argmax_map',
      output_stride: 1,
      resize_to_input: true,
      colormap: null
    },
    pack_format: 'argmax_map'
  }
};

const INPUT_DEFAULTS = {
  width: 640, height: 640, channels: 3,
  normalize: true, mean: [0.0, 0.0, 0.0], std: [1.0, 1.0, 1.0],
  scale: true, letterbox: false,
  auto_pad_color: [114, 114, 114], preserve_aspect_ratio: true,
  color_order: 'RGB',
  input_str: { layout: 'HWC', dtype: 'float32', quantized: false }
};

const RUNTIME_DEFAULTS = {
  backend: 'onnxruntime', device: 'cpu',
  threads: { intra_op: null, inter_op: null, num_threads: null },
  onnx: { providers: ['CPUExecutionProvider'], provider_options: {} },
  tflite: null,
  warmup: { runs: 0, enabled: true },
  runtimeShapes: { out_coords_space: 'tensor_pixels' }
};

// ── API pública ─────────────────────────────────────────────────────────────

export function openBuilder(builderEl, modelFile, cfgsDir, existing, onSave) {
  _builderEl = builderEl;
  _cfgsDir   = cfgsDir;
  _onSave    = onSave;

  const type = existing?.model_type || 'detection';
  _state = {
    modelFile,
    step: 1,
    config: existing ? clone(existing) : buildDefault(type)
  };

  // Asegurar que el campo out_coords_space exista en output para el builder
  if (_state.config.model_type === 'detection') {
    const rs = _state.config.runtime?.runtimeShapes;
    if (rs?.out_coords_space && !_state.config.output.out_coords_space) {
      _state.config.output.out_coords_space = rs.out_coords_space;
    }
  }

  builderEl.style.display = 'block';
  builderEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  render();
}

export function closeBuilder() {
  if (_builderEl) { _builderEl.style.display = 'none'; _builderEl.innerHTML = ''; }
  document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
  _state = null;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function buildDefault(type) {
  return {
    model_type: type,
    input:   clone(INPUT_DEFAULTS),
    output:  clone(OUTPUT_DEFAULTS[type]),
    runtime: clone(RUNTIME_DEFAULTS)
  };
}

function clone(obj) { return JSON.parse(JSON.stringify(obj)); }

// ── Render principal ─────────────────────────────────────────────────────────

function render() {
  const { modelFile, step, config } = _state;
  const baseName = modelFile.replace(/\.[^.]+$/, '');

  _builderEl.innerHTML = `
    <div class="builder-header">
      <div class="builder-title">
        <span class="builder-label">Configurando</span>
        <span class="builder-model-name">${baseName}</span>
      </div>
      <button class="close-builder-btn" id="close-builder-btn" title="Cerrar">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>

    <div class="stepper-bar">
      ${['Tipo', 'Input', 'Output', 'Runtime'].map((label, i) => {
        const n = i + 1;
        const cls = step > n ? 'done' : step === n ? 'active' : '';
        return `
          <div class="step-item ${cls}">
            <div class="step-circle">${step > n ? '✓' : n}</div>
            <span class="step-label">${label}</span>
          </div>
          ${i < 3 ? `<div class="step-line ${step > n ? 'done' : ''}"></div>` : ''}
        `;
      }).join('')}
    </div>

    <div class="step-content" id="step-content">
      ${stepContent(step, config)}
    </div>

    <div class="builder-footer">
      <button class="action-button outline-button" id="prev-btn" ${step === 1 ? 'disabled' : ''}>Anterior</button>
      <span class="step-counter">${step} / 4</span>
      <button class="action-button primary-button" id="next-btn">${step === 4 ? 'Guardar' : 'Siguiente'}</button>
    </div>
  `;

  document.getElementById('close-builder-btn').addEventListener('click', closeBuilder);
  document.getElementById('prev-btn').addEventListener('click', prevStep);
  document.getElementById('next-btn').addEventListener('click', nextStep);
  bindListeners(step);
}

// ── Contenido de cada paso ───────────────────────────────────────────────────

function stepContent(step, config) {
  switch (step) {
    case 1: return step1(config);
    case 2: return step2(config);
    case 3: return step3(config);
    case 4: return step4(config);
    default: return '';
  }
}

function step1(config) {
  const types = [
    { value: 'detection',      label: 'Detección',          desc: 'Localiza objetos con bounding boxes.' },
    { value: 'classification', label: 'Clasificación',       desc: 'Asigna una o varias clases a la imagen.' },
    { value: 'segmentation',   label: 'Segmentación semántica', desc: 'Asigna una clase a cada píxel.' }
  ];
  return `
    <div class="step-section">
      <h3 class="step-section-title">Tipo de modelo</h3>
      <div class="type-cards">
        ${types.map(t => `
          <label class="type-card ${config.model_type === t.value ? 'selected' : ''}">
            <input type="radio" name="model_type" value="${t.value}" ${config.model_type === t.value ? 'checked' : ''} hidden>
            <div class="type-card-name">${t.label}</div>
            <div class="type-card-desc">${t.desc}</div>
          </label>
        `).join('')}
      </div>
    </div>
  `;
}

function step2(config) {
  const inp  = config.input;
  const iStr = inp.input_str || {};
  const isDet = config.model_type === 'detection';
  return `
    <div class="step-section">
      <h3 class="step-section-title">Dimensiones</h3>
      <div class="form-grid-3">
        ${num('input.width',    'Ancho',   inp.width)}
        ${num('input.height',   'Alto',    inp.height)}
        ${num('input.channels', 'Canales', inp.channels)}
      </div>
      <div class="form-grid-2">
        ${sel('input.color_order',       'Orden de color',   inp.color_order,        ['RGB','BGR','GRAY'])}
        ${sel('input.input_str.layout',  'Layout del tensor', iStr.layout || 'HWC',  ['HWC','CHW','NHWC','NCHW'])}
        ${sel('input.input_str.dtype',   'Tipo de dato',     iStr.dtype  || 'float32',['float32','uint8','int8'])}
      </div>
    </div>

    <div class="step-section">
      <h3 class="step-section-title">Preprocesamiento</h3>
      <div class="form-checks">
        ${chk('input.scale',              'Escalar a [0,1] (÷255)',             inp.scale)}
        ${chk('input.normalize',          'Normalizar con media y std',         inp.normalize)}
        ${chk('input.input_str.quantized','Modelo cuantizado',                  iStr.quantized || false)}
      </div>
      <div id="normalize-fields" class="${inp.normalize ? '' : 'hidden'}">
        <div class="form-grid-label">Media (mean)</div>
        <div class="form-grid-3">
          ${numf('input.mean.0','R', inp.mean?.[0] ?? 0)}
          ${numf('input.mean.1','G', inp.mean?.[1] ?? 0)}
          ${numf('input.mean.2','B', inp.mean?.[2] ?? 0)}
        </div>
        <div class="form-grid-label">Desviación estándar (std)</div>
        <div class="form-grid-3">
          ${numf('input.std.0','R', inp.std?.[0] ?? 1)}
          ${numf('input.std.1','G', inp.std?.[1] ?? 1)}
          ${numf('input.std.2','B', inp.std?.[2] ?? 1)}
        </div>
      </div>
    </div>

    ${isDet ? `
    <div class="step-section">
      <h3 class="step-section-title">Letterbox</h3>
      <div class="form-checks">
        ${chk('input.letterbox',            'Aplicar letterbox',         inp.letterbox)}
        ${chk('input.preserve_aspect_ratio','Preservar aspect ratio',    inp.preserve_aspect_ratio)}
      </div>
      <div id="letterbox-fields" class="${inp.letterbox ? '' : 'hidden'}">
        <div class="form-grid-label">Color de padding</div>
        <div class="form-grid-3">
          ${num('input.auto_pad_color.0','R', inp.auto_pad_color?.[0] ?? 114)}
          ${num('input.auto_pad_color.1','G', inp.auto_pad_color?.[1] ?? 114)}
          ${num('input.auto_pad_color.2','B', inp.auto_pad_color?.[2] ?? 114)}
        </div>
      </div>
    </div>` : ''}
  `;
}

function step3(config) {
  switch (config.model_type) {
    case 'detection':      return step3Detection(config.output);
    case 'classification': return step3Classification(config.output);
    case 'segmentation':   return step3Segmentation(config.output);
    default: return '';
  }
}

function step3Detection(out) {
  const ts  = out.tensor_structure || {};
  const fmt = ts.box_format || 'xyxy';
  const coordKeys = { xyxy: ['x1','y1','x2','y2'], cxcywh: ['cx','cy','w','h'], yxyx: ['y1','x1','y2','x2'] };
  const keys = coordKeys[fmt] || coordKeys['xyxy'];
  const co   = ts.coordinates || {};
  return `
    <div class="step-section">
      <h3 class="step-section-title">Formato del tensor de salida</h3>
      <div class="form-grid-2">
        ${sel('output.pack_format',     'Formato de empaquetado', out.pack_format,              ['raw','yolo_flat','boxes_scores','tflite_detpost','anchor_deltas'])}
        ${sel('output.out_coords_space','Espacio de coordenadas', out.out_coords_space || 'tensor_pixels', ['tensor_pixels','normalized_0_1'])}
      </div>
    </div>
    <div class="step-section">
      <h3 class="step-section-title">Estructura por detección</h3>
      <div class="form-grid-2">
        ${sel('output.tensor_structure.box_format','Formato de boxes', fmt, ['xyxy','cxcywh','yxyx'])}
        ${num('output.tensor_structure.num_classes',       'Número de clases',    ts.num_classes ?? 80)}
      </div>
      <div class="form-grid-2">
        ${num('output.tensor_structure.confidence_index',  'Índice de confianza', ts.confidence_index ?? 6)}
        ${num('output.tensor_structure.class_index',       'Índice de clase',     ts.class_index ?? 5)}
      </div>
      <div class="form-grid-label">Índices de coordenadas</div>
      <div class="form-grid-coords" id="coords-grid">
        ${keys.map(k => num(`output.tensor_structure.coordinates.${k}`, k, co[k] ?? 0)).join('')}
      </div>
    </div>
    <div class="step-section">
      <h3 class="step-section-title">Filtrado y NMS</h3>
      <div class="form-grid-2">
        ${numf('output.confidence_threshold','Umbral de confianza', out.confidence_threshold ?? 0.5)}
        ${num('output.top_k',                'Top-K (0 = sin límite)', out.top_k ?? 0)}
      </div>
      <div class="form-checks">
        ${chk('output.apply_conf_filter','Aplicar filtro de confianza', out.apply_conf_filter ?? true)}
        ${chk('output.apply_nms',        'Aplicar NMS',                 out.apply_nms ?? false)}
        ${chk('output.nms_per_class',    'NMS por clase',               out.nms_per_class ?? false)}
      </div>
      <div id="nms-fields" class="${out.apply_nms ? '' : 'hidden'}">
        ${numf('output.nms_threshold','Umbral IoU para NMS', out.nms_threshold ?? 0.45)}
      </div>
    </div>
  `;
}

function step3Classification(out) {
  const ts = out.tensor_structure || {};
  return `
    <div class="step-section">
      <h3 class="step-section-title">Formato del tensor de salida</h3>
      <div class="form-grid-2">
        ${sel('output.pack_format',                  'Formato de salida',   out.pack_format || 'softmax_out', ['softmax_out','sigmoid_out','logits_raw'])}
        ${sel('output.tensor_structure.output_format','El modelo emite',    ts.output_format || 'logits',     ['logits','probabilities'])}
      </div>
      <div class="form-checks">
        ${chk('output.tensor_structure.multi_label','Multi-etiqueta (sigmoid por clase)', ts.multi_label || false)}
      </div>
    </div>
    <div class="step-section">
      <h3 class="step-section-title">Postprocesamiento</h3>
      <div class="form-checks">
        ${chk('output.apply_softmax','Aplicar softmax',              out.apply_softmax ?? true)}
        ${chk('output.apply_sigmoid','Aplicar sigmoid (multi-label)',out.apply_sigmoid ?? false)}
      </div>
      <div class="form-grid-2">
        ${num( 'output.tensor_structure.num_classes','Número de clases',   ts.num_classes ?? 1000)}
        ${num( 'output.top_k',                       'Top-K resultados',   out.top_k ?? 1)}
        ${numf('output.confidence_threshold',         'Umbral de confianza',out.confidence_threshold ?? 0.5)}
      </div>
      ${txt('output.label_map','Label map (path a .txt, opcional)', out.label_map || '')}
    </div>
  `;
}

function step3Segmentation(out) {
  const ts = out.tensor_structure || {};
  return `
    <div class="step-section">
      <h3 class="step-section-title">Formato del tensor de salida</h3>
      <div class="form-grid-2">
        ${sel('output.pack_format',                   'Formato de empaquetado', out.pack_format || 'argmax_map', ['argmax_map','softmax_map','binary_mask'])}
        ${sel('output.tensor_structure.output_format','El modelo emite',        ts.output_format || 'argmax_map', ['argmax_map','softmax_map'])}
      </div>
    </div>
    <div class="step-section">
      <h3 class="step-section-title">Estructura del tensor</h3>
      <div class="form-grid-2">
        ${num( 'output.tensor_structure.num_classes',   'Número de clases',            ts.num_classes ?? 21)}
        ${num( 'output.tensor_structure.output_stride', 'Output stride (factor de reducción)', ts.output_stride ?? 1)}
        ${numf('output.confidence_threshold',            'Umbral de confianza',         out.confidence_threshold ?? 0.5)}
      </div>
      <div class="form-checks">
        ${chk('output.tensor_structure.resize_to_input','Redimensionar máscara al tamaño original', ts.resize_to_input ?? true)}
      </div>
      ${txt('output.label_map','Label map (path a .txt, opcional)', out.label_map || '')}
    </div>
  `;
}

function step4(config) {
  const rt       = config.runtime || {};
  const onnx     = rt.onnx    || {};
  const tflite   = rt.tflite  || {};
  const threads  = rt.threads || {};
  const warmup   = rt.warmup  || {};
  const rs       = rt.runtimeShapes || {};
  const providers  = onnx.providers  || [];
  const delegates  = tflite.delegates|| [];
  const isDet      = config.model_type === 'detection';
  const bk         = rt.backend || 'onnxruntime';
  return `
    <div class="step-section">
      <h3 class="step-section-title">Backend e inferencia</h3>
      <div class="form-grid-2">
        ${sel('runtime.backend','Backend',     bk,              ['onnxruntime','tflite','tensorflow'])}
        ${sel('runtime.device', 'Dispositivo', rt.device || 'cpu', ['cpu','gpu'])}
        ${num('runtime.warmup.runs','Warmup runs (0 = desactivado)', warmup.runs ?? 0)}
      </div>
    </div>

    <div id="onnx-section" class="step-section ${bk !== 'onnxruntime' ? 'hidden' : ''}">
      <h3 class="step-section-title">ONNX Runtime</h3>
      <div class="form-grid-label">Providers (orden de prioridad)</div>
      <div class="form-checks">
        ${chk('runtime.onnx.providers.CUDA','CUDAExecutionProvider', providers.includes('CUDAExecutionProvider'))}
        ${chk('runtime.onnx.providers.CPU', 'CPUExecutionProvider',  providers.includes('CPUExecutionProvider') || providers.length === 0)}
      </div>
      <div class="form-grid-2">
        ${numn('runtime.threads.intra_op','Threads intra-op (vacío = auto)', threads.intra_op)}
        ${numn('runtime.threads.inter_op','Threads inter-op (vacío = auto)', threads.inter_op)}
      </div>
    </div>

    <div id="tflite-section" class="step-section ${bk !== 'tflite' ? 'hidden' : ''}">
      <h3 class="step-section-title">TFLite</h3>
      <div class="form-grid-label">Delegates</div>
      <div class="form-checks">
        ${chk('runtime.tflite.delegates.gpu','GPU delegate', delegates.includes('gpu'))}
      </div>
      ${numn('runtime.threads.num_threads','Threads (vacío = auto)', threads.num_threads)}
    </div>

    ${isDet ? `
    <div class="step-section">
      <h3 class="step-section-title">Espacio de coordenadas de salida</h3>
      ${sel('runtime.runtimeShapes.out_coords_space','Las coordenadas del tensor vienen en', rs.out_coords_space || 'tensor_pixels', ['tensor_pixels','normalized_0_1'])}
    </div>` : ''}
  `;
}

// ── Field helpers ────────────────────────────────────────────────────────────

function num(key, label, value) {
  return `<div class="control-group"><label>${label}</label>
    <input type="number" class="text-input" data-field="${key}" value="${value ?? ''}" min="0"></div>`;
}
function numf(key, label, value) {
  return `<div class="control-group"><label>${label}</label>
    <input type="number" class="text-input" data-field="${key}" value="${value ?? ''}" step="0.01"></div>`;
}
function numn(key, label, value) {
  return `<div class="control-group"><label>${label}</label>
    <input type="number" class="text-input" data-field="${key}" value="${value ?? ''}" placeholder="null" min="1"></div>`;
}
function sel(key, label, value, opts) {
  return `<div class="control-group"><label>${label}</label>
    <select class="select-control" data-field="${key}">
      ${opts.map(o => `<option value="${o}" ${o === value ? 'selected' : ''}>${o}</option>`).join('')}
    </select></div>`;
}
function chk(key, label, checked) {
  return `<label class="check-label">
    <input type="checkbox" data-field="${key}" ${checked ? 'checked' : ''}><span>${label}</span></label>`;
}
function txt(key, label, value) {
  return `<div class="control-group"><label>${label}</label>
    <input type="text" class="text-input" data-field="${key}" value="${value ?? ''}"></div>`;
}

// ── Event listeners ──────────────────────────────────────────────────────────

function bindListeners(step) {
  const content = document.getElementById('step-content');
  if (!content) return;

  // Step 1: type cards
  if (step === 1) {
    content.querySelectorAll('.type-card').forEach(card => {
      card.addEventListener('click', () => {
        const radio = card.querySelector('input[type=radio]');
        if (!radio) return;
        const newType = radio.value;
        if (newType !== _state.config.model_type) {
          _state.config.model_type = newType;
          _state.config.output = clone(OUTPUT_DEFAULTS[newType]);
        }
        content.querySelectorAll('.type-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
      });
    });
    return;
  }

  // Steps 2-4: field changes
  content.addEventListener('change', e => {
    const target = e.target;
    const key    = target.dataset.field;
    if (!key) return;

    let value;
    if (target.type === 'checkbox') {
      value = target.checked;
    } else if (target.type === 'number') {
      value = target.value === '' ? null : parseFloat(target.value);
    } else {
      value = target.value || null;
    }

    // Arrays de providers/delegates (tratamiento especial)
    if (key.startsWith('runtime.onnx.providers.')) {
      const map   = { CUDA: 'CUDAExecutionProvider', CPU: 'CPUExecutionProvider' };
      const pName = map[key.split('.').pop()];
      if (!_state.config.runtime.onnx) _state.config.runtime.onnx = { providers: [], provider_options: {} };
      const arr = _state.config.runtime.onnx.providers;
      if (value) { if (!arr.includes(pName)) arr.push(pName); }
      else       { const i = arr.indexOf(pName); if (i !== -1) arr.splice(i, 1); }
      // CUDA siempre primero
      _state.config.runtime.onnx.providers = arr.sort(a => a.includes('CUDA') ? -1 : 1);
      return;
    }
    if (key.startsWith('runtime.tflite.delegates.')) {
      const dName = key.split('.').pop();
      if (!_state.config.runtime.tflite) _state.config.runtime.tflite = { delegates: [], delegate_options: {} };
      const arr = _state.config.runtime.tflite.delegates;
      if (value) { if (!arr.includes(dName)) arr.push(dName); }
      else       { const i = arr.indexOf(dName); if (i !== -1) arr.splice(i, 1); }
      return;
    }

    setDeep(_state.config, key.split('.'), value);

    // Reactividad: mostrar/ocultar secciones condicionales
    if (key === 'input.normalize')   toggleHidden('normalize-fields', !value);
    if (key === 'input.letterbox')   toggleHidden('letterbox-fields', !value);
    if (key === 'output.apply_nms')  toggleHidden('nms-fields',       !value);
    if (key === 'runtime.backend') {
      toggleHidden('onnx-section',   value !== 'onnxruntime');
      toggleHidden('tflite-section', value !== 'tflite');
    }
    if (key === 'output.tensor_structure.box_format') {
      const coordKeys = { xyxy: ['x1','y1','x2','y2'], cxcywh: ['cx','cy','w','h'], yxyx: ['y1','x1','y2','x2'] };
      const keys = coordKeys[value] || coordKeys['xyxy'];
      _state.config.output.tensor_structure.coordinates = {};
      const grid = document.getElementById('coords-grid');
      if (grid) grid.innerHTML = keys.map(k => num(`output.tensor_structure.coordinates.${k}`, k, 0)).join('');
    }
  });
}

function toggleHidden(id, shouldHide) {
  document.getElementById(id)?.classList.toggle('hidden', shouldHide);
}

// Asigna un valor en profundidad usando una lista de partes del path
function setDeep(obj, parts, value) {
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    const next = parts[i + 1];
    if (obj[part] == null) obj[part] = isNaN(next) ? {} : [];
    obj = obj[part];
  }
  const last = parts[parts.length - 1];
  if (Array.isArray(obj)) { obj[parseInt(last)] = value; }
  else                    { obj[last] = value; }
}

// ── Navegación ───────────────────────────────────────────────────────────────

function prevStep() {
  if (_state.step > 1) { _state.step--; render(); }
}

function nextStep() {
  if (_state.step < 4) { _state.step++; render(); }
  else save();
}

// ── Guardar ──────────────────────────────────────────────────────────────────

function save() {
  const { config, modelFile } = _state;
  const baseName  = modelFile.replace(/\.[^.]+$/, '');
  const cfgPath   = path.join(_cfgsDir, baseName + '.json');
  const final     = clone(config);

  // out_coords_space vive en runtime.runtimeShapes en el schema
  if (final.output?.out_coords_space) {
    if (!final.runtime.runtimeShapes) final.runtime.runtimeShapes = {};
    final.runtime.runtimeShapes.out_coords_space = final.output.out_coords_space;
    delete final.output.out_coords_space;
  }

  // Limpiar backend no utilizado
  if (final.runtime.backend !== 'tflite')       final.runtime.tflite = null;
  if (final.runtime.backend !== 'onnxruntime')  final.runtime.onnx   = null;

  try {
    fs.writeFileSync(cfgPath, JSON.stringify(final, null, 2), 'utf-8');
    showMsg('Config guardada correctamente', 'save-ok');
    if (_onSave) _onSave();
  } catch (err) {
    showMsg(`Error al guardar: ${err.message}`, 'save-error');
  }
}

function showMsg(text, cls) {
  _builderEl.querySelector('.save-feedback')?.remove();
  const el = document.createElement('div');
  el.className   = `save-feedback ${cls}`;
  el.textContent = text;
  _builderEl.querySelector('.builder-footer').after(el);
  if (cls === 'save-ok') setTimeout(() => el.remove(), 3000);
}
