// modelsManager.js — vista "Modelos": tarjetas de pesos, dropzone e ingreso al builder.
//
// Post-hardening: este modulo NO toca el disco. Todo pasa por window.uncaAPI
// (expuesta por el preload via contextBridge), que delega en el main process.
// Por eso el escaneo y la lectura de configs son async (IPC invoke).

import { openBuilder } from './configBuilder.js';

// Extensiones aceptadas por el dropzone. Es solo filtro de UX: la validacion
// real (la que importa) la repite el main process en models:import.
const SUPPORTED_EXTENSIONS = new Set([
  '.onnx',
  '.tflite',
  '.h5',
  '.keras',
  '.pt',
  '.pth',
]);

// Helper local: extension en minusculas con punto ('.onnx'), '' si no tiene.
// (El modulo 'path' de Node ya no esta disponible en el renderer.)
function extOf(fileName) {
  const i = fileName.lastIndexOf('.');
  return i === -1 ? '' : fileName.slice(i).toLowerCase();
}

export function initModelsManager() {
  const cardsEl = document.getElementById('model-cards-grid');
  const dropzoneEl = document.getElementById('model-dropzone');
  const builderEl = document.getElementById('config-builder');
  const refreshBtn = document.getElementById('refresh-models-btn');

  scanAndRender(cardsEl, builderEl);
  initDropzone(dropzoneEl, cardsEl, builderEl);
  refreshBtn.addEventListener('click', () => scanAndRender(cardsEl, builderEl));
}

async function scanAndRender(cardsEl, builderEl) {
  cardsEl.innerHTML = '';

  // El main process escanea models/ y reporta cada peso con su estado de config
  const res = await window.uncaAPI.listModels();
  if (!res.ok) {
    cardsEl.innerHTML =
      '<p class="no-models-msg">No se pudo leer la carpeta models/</p>';
    return;
  }

  if (res.models.length === 0) {
    cardsEl.innerHTML =
      '<p class="no-models-msg">No hay modelos. Arrastrá uno a la derecha.</p>';
    return;
  }

  res.models.forEach(({ file, ext, baseName, hasConfig }) => {
    const card = document.createElement('div');
    card.className = 'model-card';
    card.dataset.file = file;
    card.innerHTML = `
      <div class="model-card-badge model-badge-${ext}">${ext.toUpperCase()}</div>
      <div class="model-card-name" title="${file}">${baseName}</div>
      <div class="model-card-meta">.${ext}</div>
      <div class="model-card-status ${hasConfig ? 'status-ok' : 'status-missing'}">
        ${
          hasConfig
            ? '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> Config'
            : '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg> Sin config'
        }
      </div>
    `;

    card.addEventListener('click', async () => {
      document
        .querySelectorAll('.model-card')
        .forEach((c) => c.classList.remove('selected'));
      card.classList.add('selected');

      // Pedir la config existente al main (null si el modelo aun no tiene);
      // si la lectura falla (JSON corrupto), se abre el builder con defaults.
      const cfgRes = await window.uncaAPI.readConfig(baseName);
      const existing = cfgRes.ok ? cfgRes.config : null;

      openBuilder(builderEl, file, existing, () => {
        scanAndRender(cardsEl, builderEl);
      });
    });

    cardsEl.appendChild(card);
  });
}

function initDropzone(dropzoneEl, cardsEl, builderEl) {
  dropzoneEl.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzoneEl.classList.add('drag-over');
  });

  dropzoneEl.addEventListener('dragleave', (e) => {
    if (!dropzoneEl.contains(e.relatedTarget)) {
      dropzoneEl.classList.remove('drag-over');
    }
  });

  dropzoneEl.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropzoneEl.classList.remove('drag-over');

    const files = Array.from(e.dataTransfer.files);
    const modelFiles = files.filter((f) =>
      SUPPORTED_EXTENSIONS.has(extOf(f.name))
    );
    const rejected = files.length - modelFiles.length;

    if (modelFiles.length === 0) {
      showFeedback(
        dropzoneEl,
        'Formato no soportado. Usá .onnx, .tflite, .h5, .keras, .pt o .pth',
        'error'
      );
      return;
    }

    // File.path no existe mas en Electron >= 32: el path real se resuelve en
    // el preload con webUtils.getPathForFile y la copia la hace el main.
    const paths = modelFiles.map((f) => window.uncaAPI.getPathForFile(f));
    const res = await window.uncaAPI.importModels(paths);

    res.errors.forEach(({ file, error }) =>
      console.error(`Error al copiar ${file}:`, error)
    );

    const label =
      res.copied === 1
        ? `"${modelFiles[0].name}" agregado`
        : `${res.copied} modelos agregados`;
    showFeedback(
      dropzoneEl,
      label + (rejected ? ` (${rejected} ignorado/s)` : ''),
      res.copied > 0 ? 'success' : 'error'
    );
    scanAndRender(cardsEl, builderEl);
  });
}

function showFeedback(dropzoneEl, msg, type) {
  const el = dropzoneEl.querySelector('.dropzone-feedback');
  if (!el) return;
  el.textContent = msg;
  el.className = `dropzone-feedback feedback-${type}`;
  setTimeout(() => {
    el.textContent = '';
    el.className = 'dropzone-feedback';
  }, 3500);
}
