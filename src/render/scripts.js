import { switchCamera } from './modules/cameraSwitcher.js';
import { startRecording, stopRecording, isRecording } from './modules/record.js';
import { getModels } from './modules/modelLoader.js';
import { selectModel } from './modules/selectModel.js';
import { confidenceUrl, colorsUrl, inferenceLogsUrl } from './modules/constants.js';
const d = document;

d.addEventListener('DOMContentLoaded', () => {
  const darkModeToggle    = d.getElementById('dark-mode-toggle');
  const tabButtons        = d.querySelectorAll(".tab-button");
  const tabContents       = d.querySelectorAll(".tab-content");
  const confidenceSlider  = d.getElementById("confidence-slider");
  const confidenceValue   = d.getElementById("confidence-value");
  const advancedSettingsBtn = d.getElementById("advanced-settings-btn");
  const settingsModal     = d.getElementById("settings-modal");
  const closeModalBtn     = d.getElementById("close-modal");
  const saveSettingsBtn   = d.getElementById("save-settings");
  const bboxColorInput    = d.getElementById("bbox-color");
  const labelColorInput   = d.getElementById("label-color");
  const bboxColorPreview  = d.getElementById("bbox-color-preview");
  const labelColorPreview = d.getElementById("label-color-preview");
  const cameraSelect      = d.getElementById('camera-select');
  const recordButton      = d.getElementById('record-btn');
  const video             = d.getElementById('video');
  const fileButton        = d.getElementById('personalized-upload');
  const inputFile         = d.getElementById('file-upload');
  const modelSelect       = d.getElementById('ia-model');
  const videoContainer    = d.querySelector('.video-container');
  const logToggleBtn      = d.getElementById('log-toggle-btn');
  const logPanel          = d.getElementById('log-panel');
  const logList           = d.getElementById('log-list');

  let logPollingInterval = null;

  bboxColorPreview.style.backgroundColor = bboxColorInput.value;
  labelColorPreview.style.backgroundColor = labelColorInput.value;
  getModels();

  // ── Tema oscuro ──────────────────────────────────────────────────────────
  darkModeToggle.addEventListener("change", function () {
    document.documentElement.classList.toggle("dark", this.checked);
  });

  // ── Tabs ─────────────────────────────────────────────────────────────────
  tabButtons.forEach((button) => {
    button.addEventListener("click", function () {
      const tabName = this.getAttribute("data-tab");
      tabButtons.forEach((btn) => btn.classList.remove("active"));
      this.classList.add("active");
      tabContents.forEach((content) => {
        content.classList.remove("active");
        if (content.id === `${tabName}-tab`) content.classList.add("active");
      });
    });
  });

  // ── Confianza ─────────────────────────────────────────────────────────────
  confidenceSlider.addEventListener("input", function () {
    confidenceValue.textContent = `${this.value}%`;
  });

  confidenceSlider.addEventListener("change", async function () {
    const value = parseFloat(this.value) / 100;
    try {
      await fetch(confidenceUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value }),
      });
    } catch (err) {
      console.error("Error al actualizar confianza:", err);
    }
  });

  // ── Pantalla completa (doble clic en el area de video) ───────────────────
  videoContainer.addEventListener('dblclick', () => {
    if (!document.fullscreenElement) {
      videoContainer.requestFullscreen().catch(err => {
        console.error('Error al entrar en pantalla completa:', err);
      });
    } else {
      document.exitFullscreen();
    }
  });

  // ── Modal de configuracion avanzada ──────────────────────────────────────
  advancedSettingsBtn.addEventListener("click", () => {
    settingsModal.classList.add("active");
  });

  closeModalBtn.addEventListener("click", () => {
    settingsModal.classList.remove("active");
  });

  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) settingsModal.classList.remove("active");
  });

  bboxColorInput.addEventListener("input", function () {
    bboxColorPreview.style.backgroundColor = this.value;
  });

  labelColorInput.addEventListener("input", function () {
    labelColorPreview.style.backgroundColor = this.value;
  });

  // Guardar colores → envia al backend para que los use en _draw_detections
  saveSettingsBtn.addEventListener("click", async () => {
    try {
      await fetch(colorsUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bbox_color:  bboxColorInput.value,
          label_color: labelColorInput.value,
        }),
      });
    } catch (err) {
      console.error("Error al guardar configuracion de colores:", err);
    }
    settingsModal.classList.remove("active");
  });

  // ── Selector de modelo ────────────────────────────────────────────────────
  modelSelect.addEventListener('change', () => {
    selectModel(modelSelect.value);
  });

  // ── Cargar video desde archivo ────────────────────────────────────────────
  fileButton.addEventListener('click', () => {
    inputFile.click();
  });

  // ── Camara ────────────────────────────────────────────────────────────────
  switchCamera(cameraSelect);

  // ── Grabacion (usa la flag exportada para evitar la comparacion por texto) ─
  recordButton.addEventListener('click', () => {
    if (!isRecording) {
      startRecording(recordButton, video);
    } else {
      stopRecording(recordButton);
    }
  });

  // ── Registro de errores de inferencia ────────────────────────────────────
  logToggleBtn.addEventListener('click', () => {
    const isOpen = logPanel.classList.toggle('active');
    if (isOpen) {
      fetchInferenceLogs();
      logPollingInterval = setInterval(fetchInferenceLogs, 5000);
    } else {
      clearInterval(logPollingInterval);
      logPollingInterval = null;
    }
  });

  async function fetchInferenceLogs() {
    try {
      const res = await fetch(inferenceLogsUrl);
      const { logs } = await res.json();
      if (!logs || logs.length === 0) {
        logList.innerHTML = '<li class="log-empty">Sin errores registrados.</li>';
      } else {
        logList.innerHTML = logs.slice().reverse().map(entry => `
          <li class="log-item">
            <span class="log-timestamp">${entry.timestamp}</span>
            ${entry.error}
          </li>
        `).join('');
      }
    } catch (err) {
      console.error("Error al obtener logs:", err);
    }
  }
});
