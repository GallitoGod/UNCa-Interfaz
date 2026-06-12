// ipc-handlers.js — operaciones de disco del lado del MAIN process.
//
// Todo acceso a fs vive aca (no en el renderer): el renderer pide por IPC y
// este modulo valida y ejecuta. Asi un renderer comprometido (ej: XSS en la UI)
// no puede tocar el disco fuera de las operaciones explicitamente permitidas.
//
// Cada handler devuelve siempre un objeto { ok, ... } en vez de tirar: los
// errores cruzan el IPC como datos y la UI decide como mostrarlos.

const { ipcMain, app } = require('electron');
const fs = require('fs');
const path = require('path');

// Mismo set que MODEL_EXTENSIONS en mainAPI.py — mantener sincronizados
const SUPPORTED_EXTENSIONS = new Set([
  '.onnx',
  '.tflite',
  '.h5',
  '.keras',
  '.pt',
  '.pth',
]);

// Raiz del proyecto: app.getAppPath() apunta al directorio del package.json,
// independiente del cwd desde el que se haya lanzado Electron.
const ROOT_DIR = app.getAppPath();
const MODELS_DIR = path.join(ROOT_DIR, 'models');
const CONFIGS_DIR = path.join(ROOT_DIR, 'configs');

// Guarda anti path-traversal: el baseName que manda el renderer se usa para
// armar configs/<baseName>.json, asi que no puede contener separadores ni "..".
// Devuelve true solo si es un nombre "plano" valido.
function isSafeBaseName(name) {
  return (
    typeof name === 'string' &&
    name.length > 0 &&
    name.length < 256 &&
    !name.includes('/') &&
    !name.includes('\\') &&
    !name.includes('..') &&
    name !== '.'
  );
}

function registerIpcHandlers() {
  // ── models:list ───────────────────────────────────────────────────────────
  // Escanea models/ y devuelve, por cada peso soportado, si tiene config.
  ipcMain.handle('models:list', () => {
    try {
      const files = fs
        .readdirSync(MODELS_DIR)
        .filter((f) => SUPPORTED_EXTENSIONS.has(path.extname(f).toLowerCase()));

      const models = files.map((file) => {
        const ext = path.extname(file).toLowerCase().slice(1);
        const baseName = path.basename(file, path.extname(file));
        const hasConfig = fs.existsSync(
          path.join(CONFIGS_DIR, baseName + '.json')
        );
        return { file, ext, baseName, hasConfig };
      });

      return { ok: true, models };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  });

  // ── models:import ─────────────────────────────────────────────────────────
  // Copia archivos (por path absoluto, resuelto en el preload con webUtils)
  // a models/. Revalida la extension aca: el filtro del renderer es solo UX,
  // la garantia real esta en el main.
  ipcMain.handle('models:import', (_event, paths) => {
    if (!Array.isArray(paths)) {
      return {
        ok: false,
        copied: 0,
        errors: [{ file: '', error: 'payload invalido' }],
      };
    }

    let copied = 0;
    const errors = [];
    for (const srcPath of paths) {
      const name = path.basename(srcPath);
      if (!SUPPORTED_EXTENSIONS.has(path.extname(name).toLowerCase())) {
        errors.push({ file: name, error: 'extension no soportada' });
        continue;
      }
      try {
        fs.copyFileSync(srcPath, path.join(MODELS_DIR, name));
        copied++;
      } catch (err) {
        errors.push({ file: name, error: err.message });
      }
    }
    return { ok: errors.length === 0, copied, errors };
  });

  // ── configs:read ──────────────────────────────────────────────────────────
  // Devuelve el JSON parseado de configs/<baseName>.json, o config:null si
  // no existe (el builder arranca con defaults en ese caso).
  ipcMain.handle('configs:read', (_event, baseName) => {
    if (!isSafeBaseName(baseName)) {
      return { ok: false, error: `nombre de config invalido: "${baseName}"` };
    }
    const cfgPath = path.join(CONFIGS_DIR, baseName + '.json');
    try {
      if (!fs.existsSync(cfgPath)) {
        return { ok: true, config: null };
      }
      const config = JSON.parse(fs.readFileSync(cfgPath, 'utf-8'));
      return { ok: true, config };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  });

  // ── configs:write ─────────────────────────────────────────────────────────
  // Serializa y escribe configs/<baseName>.json. El objeto llega ya clonado
  // por el IPC (structured clone), asi que no comparte referencias con la UI.
  ipcMain.handle('configs:write', (_event, baseName, config) => {
    if (!isSafeBaseName(baseName)) {
      return { ok: false, error: `nombre de config invalido: "${baseName}"` };
    }
    if (config === null || typeof config !== 'object') {
      return { ok: false, error: 'la config debe ser un objeto JSON' };
    }
    const cfgPath = path.join(CONFIGS_DIR, baseName + '.json');
    try {
      fs.writeFileSync(cfgPath, JSON.stringify(config, null, 2), 'utf-8');
      return { ok: true };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  });
}

module.exports = { registerIpcHandlers };
