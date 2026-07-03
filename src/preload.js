// preload.js — puente seguro entre el renderer aislado y el main process.
//
// Con contextIsolation activado el renderer NO tiene acceso a Node (require, fs,
// process, etc.). Este preload corre en un contexto privilegiado intermedio y seria
// el unico lugar donde exponer, via contextBridge, las operaciones IPC necesarias.
//
// El frontend es un thin client SIN acceso a disco (ver SDD): toda la persistencia
// (modelos y configs) va por el backend HTTP, asi que hoy NO se expone NINGUNA API
// de disco al renderer (antes existia window.uncaAPI con listModels/readConfig/
// importModels/writeConfig/getPathForFile — todo eliminado).
//
// El archivo se mantiene porque BrowserWindow lo referencia (webPreferences.preload)
// y como punto de extension: si en el futuro se necesita IPC NO-disco, exponerlo aca
// devolviendo siempre { success, data?, error? } (SDD 4.1.1).

// Sin exposiciones por ahora.
