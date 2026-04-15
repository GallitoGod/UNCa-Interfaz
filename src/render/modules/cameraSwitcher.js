import { initVideoStream } from "./streamHandler.js";

let currentStream = null;
let activeWebSocket = null;

export async function switchCamera(cameraSelect) {
  try {
    await navigator.mediaDevices.getUserMedia({ video: true });
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');

    if (!videoDevices.length) {
      console.error('No se detectaron camaras');
      return;
    }

    populateCameraSelect(cameraSelect, videoDevices);

    cameraSelect.addEventListener('change', async (event) => {
      await startSelectedCamera(event.target.value);
    });

    await startSelectedCamera(videoDevices[0].deviceId);

  } catch (err) {
    console.error('Error inicializando camaras:', err);
  }
}

async function startSelectedCamera(deviceId) {
  try {
    // Cerrar WebSocket anterior antes de abrir uno nuevo
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

    // Ocultar el mensaje de "sin video" cuando el stream arranca
    const noVideoMsg = document.querySelector('.no-video-message');
    if (noVideoMsg) noVideoMsg.style.display = 'none';

    activeWebSocket = initVideoStream(video);

  } catch (err) {
    const noVideoMsg = document.querySelector('.no-video-message');
    if (noVideoMsg) noVideoMsg.style.display = '';
    console.error('Error al cambiar camara:', err);
  }
}

function populateCameraSelect(selectElement, devices) {
  selectElement.innerHTML = devices.map((device, index) => `
    <option value="${device.deviceId}">
      ${device.label || `Camara ${index + 1}`}
    </option>
  `).join('');
}
