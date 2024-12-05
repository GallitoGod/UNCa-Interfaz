import { switchCamera } from './camera.js';
import { startRecording, stopRecording } from './record.js';
import { enableDarkMode, disableDarkMode } from './toggleTheme.js';

document.addEventListener('DOMContentLoaded', () => {
  const cameraSelect = document.getElementById('camera-select');
  const recordButton = document.getElementById('record-btn');
  const video = document.getElementById('video');
  const toggle = document.getElementById('dark-mode-toggle');

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
// Tambien hay que intentar poner modo oscuro y modo claro que se pueda cambiar switch.
