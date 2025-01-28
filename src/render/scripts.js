import { switchCamera } from './modules/camera.js';
import { startRecording, stopRecording } from './modules/record.js';
import { enableDarkMode, disableDarkMode } from './modules/toggleTheme.js';
import { setupFrameProcessor } from './modules/frameProcessor.js';
import { setupModelLoader } from './modules/modelLoader.js';

document.addEventListener('DOMContentLoaded', () => {
  const cameraSelect = document.getElementById('camera-select');
  const recordButton = document.getElementById('record-btn');
  const video = document.getElementById('video');
  const toggle = document.getElementById('dark-mode-toggle');
  const fileButton = document.getElementById('personalized-upload');
  const inputFile = document.getElementById('file-upload');

  setupFrameProcessor();
  setupModelLoader();
  
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
