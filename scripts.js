import { switchCamera } from './camera.js';
import { startRecording, stopRecording } from './record.js';

document.addEventListener('DOMContentLoaded', () => {
  const cameraSelect = document.getElementById('camera-select');
  const recordButton = document.getElementById('record-btn');
  const video = document.getElementById('video');

  switchCamera(cameraSelect);

  recordButton.addEventListener('click', () => {
    if (recordButton.textContent === 'Iniciar') {
      startRecording(recordButton, video);
    } else {
      stopRecording(recordButton);
    }
  });
});
