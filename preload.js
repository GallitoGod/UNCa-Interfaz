const { contextBridge, ipcRenderer } = require('electron');

console.log('Preload.js cargado');

contextBridge.exposeInMainWorld('api', {
    getModels: () => ipcRenderer.invoke('get-models'),
});
