import { switchCamera, startCamera } from './camera.js'; 

document.addEventListener('DOMContentLoaded', () => {
  const cameraSelect = document.getElementById('camera-select');
  switchCamera(cameraSelect, startCamera);
  
});
