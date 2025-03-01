import { switchCamera } from './modules/camera.js';
import { startRecording, stopRecording } from './modules/record.js';
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
  const darkModeToggle = d.getElementById('dark-mode-toggle');
  const tabButtons = document.querySelectorAll(".tab-button");
  const tabContents = document.querySelectorAll(".tab-content");
  const confidenceSlider = document.getElementById("confidence-slider");
  const confidenceValue = document.getElementById("confidence-value");
  const advancedSettingsBtn = document.getElementById("advanced-settings-btn");
  const settingsModal = document.getElementById("settings-modal");
  const closeModalBtn = document.getElementById("close-modal");
  const saveSettingsBtn = document.getElementById("save-settings");
  const bboxColorInput = document.getElementById("bbox-color");
  const labelColorInput = document.getElementById("label-color");
  const bboxColorPreview = document.getElementById("bbox-color-preview");
  const labelColorPreview = document.getElementById("label-color-preview");
  const cameraSelect = d.getElementById('camera-select');
  const recordButton = d.getElementById('record-btn');
  const video = d.getElementById('video');
  const fileButton = d.getElementById('personalized-upload');
  const inputFile = d.getElementById('file-upload');
  const modelSelect = d.getElementById('ia-model');

  bboxColorPreview.style.backgroundColor = bboxColorInput.value;
  labelColorPreview.style.backgroundColor = labelColorInput.value;
  getModels();

  darkModeToggle.addEventListener("change", function () {
    document.documentElement.classList.toggle("dark", this.checked);
  });

  tabButtons.forEach((button) => {
    button.addEventListener("click", function () {
      const tabName = this.getAttribute("data-tab");

      tabButtons.forEach((btn) => btn.classList.remove("active"));
      this.classList.add("active");

      tabContents.forEach((content) => {
        content.classList.remove("active");
        if (content.id === `${tabName}-tab`) {
          content.classList.add("active");
        }
      });
    });
  });

  confidenceSlider.addEventListener("input", function () {
    confidenceValue.textContent = `${this.value}%`;
  });

  advancedSettingsBtn.addEventListener("click", () => {
    settingsModal.classList.add("active");
  });

  closeModalBtn.addEventListener("click", () => {
    settingsModal.classList.remove("active");
  });

  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) {
      settingsModal.classList.remove("active");
    }
  });

  bboxColorInput.addEventListener("input", function () {
    bboxColorPreview.style.backgroundColor = this.value;
  });

  labelColorInput.addEventListener("input", function () {
    labelColorPreview.style.backgroundColor = this.value;
  });

  saveSettingsBtn.addEventListener("click", () => {
    settingsModal.classList.remove("active");
  });

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
});
