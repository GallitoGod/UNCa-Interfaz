import { switchCamera } from './modules/camera.js';
import { startRecording, stopRecording } from './modules/record.js';
import { enableDarkMode, disableDarkMode } from './modules/uiManager.js';
import { getModels } from './modules/modelLoader.js';
const d = document;

d.addEventListener('DOMContentLoaded', () => {
  const cameraSelect = d.getElementById('camera-select');
  const recordButton = d.getElementById('record-btn');
  const video = d.getElementById('video');
  const toggle = d.getElementById('dark-mode-toggle');
  const fileButton = d.getElementById('personalized-upload');
  const inputFile = d.getElementById('file-upload');

  getModels();
  
  fileButton.addEventListener('click', () => {
    inputFile.click();
  })
  switchCamera(cameraSelect);

  recordButton.addEventListener('click', () => {
    if (recordButton.textContent === 'Iniciar') {
      startRecording(recordButton, video);
    } else {
      stopRecording(recordButton);
    }
  });

  toggle.addEventListener('change', () => {
    if (toggle.checked) {
      enableDarkMode();
    } else {
      disableDarkMode();
    }
  });
});

// Estaria bueno hacer que si se toca el video, tome el tamaño de la ventana completa.
