// preload.js — puente seguro entre el renderer (aislado) y el proceso principal.
//
// Con contextIsolation activado el renderer NO tiene acceso a Node (require, fs,
// process, etc.). Este preload corre en un contexto privilegiado intermedio y
// expone via contextBridge SOLO las operaciones que la UI necesita, como
// funciones async que delegan en el main process por IPC (ipcRenderer.invoke).
//
// Regla de oro: aca no se implementa logica de archivos; solo se reenvia al
// main process, que es quien valida y ejecuta (ver ipc-handlers.js).

const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('uncaAPI', {
  // Lista los archivos de pesos en models/ con su estado de config.
  // Devuelve { ok, models: [{file, ext, baseName, hasConfig}] } o { ok:false, error }.
  listModels: () => ipcRenderer.invoke('models:list'),

  // Copia archivos de modelo (por path absoluto) a models/.
  // Devuelve { ok, copied, errors: [{file, error}] }.
  importModels: (paths) => ipcRenderer.invoke('models:import', paths),

  // Lee configs/<baseName>.json parseado, o null si no existe.
  // Devuelve { ok, config } o { ok:false, error }.
  readConfig: (baseName) => ipcRenderer.invoke('configs:read', baseName),

  // Escribe configs/<baseName>.json (el main valida el nombre y serializa).
  // Devuelve { ok } o { ok:false, error }.
  writeConfig: (baseName, config) =>
    ipcRenderer.invoke('configs:write', baseName, config),

  // Obtiene el path real de un File arrastrado al dropzone. Electron >= 32
  // elimino File.path del renderer; webUtils.getPathForFile es el reemplazo
  // oficial y solo esta disponible en el preload.
  getPathForFile: (file) => webUtils.getPathForFile(file),
});
