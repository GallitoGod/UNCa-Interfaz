const { app, BrowserWindow, ipcMain } = require('electron');
const fs = require('fs');
const path = require('path');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: true,
      contextIsolation: true,
      enableRemoteModule: false,
    },
  });

  mainWindow.loadFile('./static/index.html');

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

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

//Esta api va a dejar de existir para hacer APIs unicamente en python
ipcMain.handle('get-models', async () => {
  try {
    const modelsDir = path.join(__dirname, 'models');
    if (!fs.existsSync(modelsDir)) {
      throw new Error(`El directorio ${modelsDir} no existe`);
    }
  return fs.readdirSync(modelsDir).filter((file) => {
    const ext = path.extname(file).toLowerCase();
    return ext === '.onnx' || ext === '.json';
  });
  } catch (err) {
    console.error('Error al obtener los modelos:', err);
    throw err;
  }
});
