import { openBuilder } from './configBuilder.js';

const fs   = require('fs');
const path = require('path');

const SUPPORTED_EXTENSIONS = new Set(['.onnx', '.tflite', '.h5', '.keras']);
const MODELS_DIR  = path.join(process.cwd(), 'models');
const CONFIGS_DIR = path.join(process.cwd(), 'configs');

export function initModelsManager() {
  const cardsEl    = document.getElementById('model-cards-grid');
  const dropzoneEl = document.getElementById('model-dropzone');
  const builderEl  = document.getElementById('config-builder');
  const refreshBtn = document.getElementById('refresh-models-btn');

  scanAndRender(cardsEl, builderEl);
  initDropzone(dropzoneEl, cardsEl, builderEl);
  refreshBtn.addEventListener('click', () => scanAndRender(cardsEl, builderEl));
}

function scanAndRender(cardsEl, builderEl) {
  cardsEl.innerHTML = '';

  let files;
  try {
    files = fs.readdirSync(MODELS_DIR).filter(f =>
      SUPPORTED_EXTENSIONS.has(path.extname(f).toLowerCase())
    );
  } catch {
    cardsEl.innerHTML = '<p class="no-models-msg">No se pudo leer la carpeta models/</p>';
    return;
  }

  if (files.length === 0) {
    cardsEl.innerHTML = '<p class="no-models-msg">No hay modelos. Arrastrá uno a la derecha.</p>';
    return;
  }

  files.forEach(file => {
    const ext      = path.extname(file).toLowerCase().slice(1);
    const baseName = path.basename(file, path.extname(file));
    const cfgPath  = path.join(CONFIGS_DIR, baseName + '.json');
    const hasCfg   = fs.existsSync(cfgPath);

    const card = document.createElement('div');
    card.className    = 'model-card';
    card.dataset.file = file;
    card.innerHTML = `
      <div class="model-card-badge model-badge-${ext}">${ext.toUpperCase()}</div>
      <div class="model-card-name" title="${file}">${baseName}</div>
      <div class="model-card-meta">.${ext}</div>
      <div class="model-card-status ${hasCfg ? 'status-ok' : 'status-missing'}">
        ${hasCfg
          ? '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> Config'
          : '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg> Sin config'
        }
      </div>
    `;

    card.addEventListener('click', () => {
      document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');

      const existing = hasCfg
        ? JSON.parse(fs.readFileSync(cfgPath, 'utf-8'))
        : null;

      openBuilder(builderEl, file, CONFIGS_DIR, existing, () => {
        scanAndRender(cardsEl, builderEl);
      });
    });

    cardsEl.appendChild(card);
  });
}

function initDropzone(dropzoneEl, cardsEl, builderEl) {
  dropzoneEl.addEventListener('dragover', e => {
    e.preventDefault();
    dropzoneEl.classList.add('drag-over');
  });

  dropzoneEl.addEventListener('dragleave', e => {
    if (!dropzoneEl.contains(e.relatedTarget)) {
      dropzoneEl.classList.remove('drag-over');
    }
  });

  dropzoneEl.addEventListener('drop', e => {
    e.preventDefault();
    dropzoneEl.classList.remove('drag-over');

    const files      = Array.from(e.dataTransfer.files);
    const modelFiles = files.filter(f =>
      SUPPORTED_EXTENSIONS.has(path.extname(f.name).toLowerCase())
    );
    const rejected   = files.length - modelFiles.length;

    if (modelFiles.length === 0) {
      showFeedback(dropzoneEl, 'Formato no soportado. Usá .onnx, .tflite o .keras', 'error');
      return;
    }

    let copied = 0;
    modelFiles.forEach(f => {
      try {
        fs.copyFileSync(f.path, path.join(MODELS_DIR, f.name));
        copied++;
      } catch (err) {
        console.error(`Error al copiar ${f.name}:`, err);
      }
    });

    const label = copied === 1 ? `"${modelFiles[0].name}" agregado` : `${copied} modelos agregados`;
    showFeedback(dropzoneEl, label + (rejected ? ` (${rejected} ignorado/s)` : ''), 'success');
    scanAndRender(cardsEl, builderEl);
  });
}

function showFeedback(dropzoneEl, msg, type) {
  const el = dropzoneEl.querySelector('.dropzone-feedback');
  if (!el) return;
  el.textContent  = msg;
  el.className    = `dropzone-feedback feedback-${type}`;
  setTimeout(() => {
    el.textContent = '';
    el.className   = 'dropzone-feedback';
  }, 3500);
}
