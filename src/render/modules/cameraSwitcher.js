import { initVideoStream } from "./streamHandler.js";

let currentStream = null;
let activeWebSocket = null;

export async function initCameras(cameraSelect) {
  try {
    await navigator.mediaDevices.getUserMedia({ video: true });
    await _populateCameraSelect(cameraSelect);

    cameraSelect.addEventListener('change', async (event) => {
      await startSelectedCamera(event.target.value);
    });

    if (cameraSelect.value) {
      await startSelectedCamera(cameraSelect.value);
    }
  } catch (err) {
    console.error('Error inicializando camaras:', err);
  }
}

export async function refreshCameras(cameraSelect) {
  try {
    const previousId = cameraSelect.value;
    await _populateCameraSelect(cameraSelect);

    // Si la camara que estaba seleccionada sigue disponible, mantenerla
    const stillAvailable = [...cameraSelect.options].some(o => o.value === previousId);
    if (stillAvailable && previousId) {
      cameraSelect.value = previousId;
    }

    if (cameraSelect.value) {
      await startSelectedCamera(cameraSelect.value);
    }
  } catch (err) {
    console.error('Error al actualizar camaras:', err);
  }
}

async function _populateCameraSelect(cameraSelect) {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoDevices = devices.filter(device => device.kind === 'videoinput');

  if (!videoDevices.length) {
    console.error('No se detectaron camaras');
    return;
  }

  const currentValue = cameraSelect.value;
  cameraSelect.innerHTML = videoDevices.map((device, index) => `
    <option value="${device.deviceId}">
      ${device.label || `Camara ${index + 1}`}
    </option>
  `).join('');

  // Restaurar seleccion si sigue existiendo
  const stillExists = [...cameraSelect.options].some(o => o.value === currentValue);
  if (stillExists) cameraSelect.value = currentValue;
}

export async function startSelectedCamera(deviceId) {
  try {
    if (activeWebSocket) {
      activeWebSocket.close();
      activeWebSocket = null;
    }
    if (currentStream) {
      currentStream.getTracks().forEach(track => track.stop());
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: { exact: deviceId },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 }
      }
    });

    const video = document.getElementById('video');
    video.srcObject = stream;
    currentStream = stream;
    await video.play();

    const noVideoMsg = document.querySelector('.no-video-message');
    if (noVideoMsg) noVideoMsg.style.display = 'none';

    activeWebSocket = initVideoStream(video);

  } catch (err) {
    const noVideoMsg = document.querySelector('.no-video-message');
    if (noVideoMsg) noVideoMsg.style.display = '';
    console.error('Error al cambiar camara:', err);
  }
}

export function stopCurrentStream() {
  if (activeWebSocket) {
    activeWebSocket.close();
    activeWebSocket = null;
  }
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
    currentStream = null;
  }
  const video = document.getElementById('video');
  if (video) video.srcObject = null;
}
