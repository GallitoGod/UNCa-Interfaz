// main.js — proceso principal de Electron.
// Crea la ventana con la configuracion de seguridad recomendada, registra los
// handlers IPC (hoy no-op: thin client sin disco) y arranca/detiene el backend
// uvicorn (backend-process.js).
//
// Carga el frontend React/Vite (client/):
//   - dev (electron . --dev): el dev-server de Vite (http://localhost:5173) con HMR.
//   - prod (electron .):      el build estatico en client/dist/index.html.

const { app, BrowserWindow } = require('electron');
const path = require('path');
const { registerIpcHandlers } = require('./ipc-handlers');
const { startBackend, stopBackend } = require('./backend-process');

// --dev usa el dev-server de Vite (correr `npm run dev` en otra terminal).
const IS_DEV = process.argv.includes('--dev');
const DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';

// Raiz del repo (main.js vive en src/). El backend se arranca con cwd = raiz.
const PROJECT_ROOT = path.join(__dirname, '..');

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
  if (IS_DEV) {
    mainWindow.loadURL(DEV_SERVER_URL);
    mainWindow.webContents.openDevTools();
  } else {
    // base relativa en vite.config.mts -> assets con rutas ./, validas bajo file://
    mainWindow.loadFile(path.join(__dirname, '..', 'client', 'dist', 'index.html'));
  }

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  // Los handlers se registran una sola vez, antes de crear la ventana,
  // para que el renderer nunca invoque un canal todavia inexistente.
  registerIpcHandlers();
  // Arranca uvicorn en paralelo: el renderer reintenta sus fetch hasta que el
  // backend responde, asi no hace falta bloquear la creacion de la ventana.
  // Escape UNCA_NO_SPAWN=1 para desarrollo con el backend a mano en otra terminal.
  startBackend({ projectRoot: PROJECT_ROOT });
  createWindow();
});

// Matar el backend pase lo que pase al salir (cierre normal o por la ventana).
app.on('before-quit', function () {
  stopBackend();
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
