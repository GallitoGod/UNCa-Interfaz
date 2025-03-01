import { switchCamera } from './modules/camera.js';
import { startRecording, stopRecording } from './modules/record.js';
import { enableDarkMode, disableDarkMode } from './modules/uiManager.js';
import { getModels } from './modules/modelLoader.js';
import { selectModel } from './modules/selectModel.js';
import { selectedModel } from './modules/constants.js';
const d = document;

/*
  Personalizacion de colores y estilos desde el cliente:
    Crear un menu de configuracion donde se puedan cambiar los color, grosor, etiquetas, etc. Luego,
  enviar estos valores a la API junto con cada imagen.
    Cuando se cambian los estilos, por nada en el mundo cambiar la IA, solo los estilos. Tendria que buscar alguna
  forma de hacer mas dicifil el cambio de la IA en medio de las predicciones.

    IDEAS:
      - Podria usar un input color en el HTML para elegirlo dinamicamente:
        <input type="color" id="color-picker">
      - Cambiar el color y enviarlo con cada prediccion:
        let color = document.getElementById('color-picker').value;
        let rgbcColor = hexToRgb(color);
        function hexToRgb(hex) {
          let bigint = parseInt(hex.substring(1), 16);
          let r = (bigint >> 16) & 255;
          let g = (bigint >> 8) & 255;
          let b = bigint & 255;
          return `${r},${g},${b}`;       // ALGO ASI PODRIA SER
        }
      - Podria usar localStorage para guardar las configuraciones de usurio, si es que electron puede claro.
*/

d.addEventListener('DOMContentLoaded', () => {
  const cameraSelect = d.getElementById('camera-select');
  const recordButton = d.getElementById('record-btn');
  const video = d.getElementById('video');
  const toggle = d.getElementById('dark-mode-toggle');
  const fileButton = d.getElementById('personalized-upload');
  const inputFile = d.getElementById('file-upload');
  const modelSelect = d.getElementById('ia-model');

  getModels();

  modelSelect.addEventListener('Change', () => {
    selectedModel = modelSelect.value;
    selectModel();
  });
  
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
