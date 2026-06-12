// main.js — proceso principal de Electron.
// Crea la ventana con la configuracion de seguridad recomendada y registra
// los handlers IPC que concentran TODO el acceso a disco (ipc-handlers.js).

const { app, BrowserWindow } = require('electron');
const path = require('path');
const { registerIpcHandlers } = require('./ipc-handlers');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 1000,
    webPreferences: {
      // Hardening (antes: nodeIntegration:true + contextIsolation:false):
      // - nodeIntegration:false  -> el renderer no tiene require/fs/process.
      // - contextIsolation:true  -> el preload corre en un mundo JS separado;
      //   solo lo expuesto via contextBridge llega a la pagina (window.uncaAPI).
      // - sandbox:true           -> el renderer corre sandboxeado a nivel OS.
      // El acceso a disco vive en el main process detras de IPC validado.
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });
  mainWindow.loadFile('./static/index.html');
  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  // Los handlers se registran una sola vez, antes de crear la ventana,
  // para que el renderer nunca invoque un canal todavia inexistente.
  registerIpcHandlers();
  createWindow();
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', function () {
  if (mainWindow === null) {
    createWindow();
  }
});
