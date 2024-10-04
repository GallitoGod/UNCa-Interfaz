const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const fs = require('fs');
const path = require('path');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  mainWindow.loadFile('index.html');

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

// Manejar la carga de imágenes
ipcMain.on('upload-btn', async (event) => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'Images', extensions: ['jpg', 'png', 'jpeg'] }]
  });

  if (!result.canceled) {
    const imagePath = result.filePaths[0];
    const imageBuffer = fs.readFileSync(imagePath);
    event.reply('image-loaded', { imagePath, imageBuffer });
  } else {
    event.reply('image-load-cancelled');
  }
});

// Procesar la imagen con un modelo de IA (ejemplo simple)
ipcMain.on('process-image', (event, { imagePath, modelName }) => {
  // Aquí puedes realizar el análisis de la imagen con el modelo IA
  // Puedes utilizar bibliotecas como TensorFlow.js o alguna que hayas entrenado
  console.log(`Procesando imagen: ${imagePath} con el modelo: ${modelName}`);

  // Ejemplo de análisis de IA simulado:
  const analysisResults = `La imagen ${path.basename(imagePath)} fue procesada con el modelo ${modelName}.`;

  // Enviar los resultados al frontend
  event.reply('analysis-complete', analysisResults);
});