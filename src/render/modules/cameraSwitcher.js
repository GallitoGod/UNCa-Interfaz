import { initVideoStream } from "./streamHandler.js";

let currentStream = null;
let activeWebSocket = null;

export async function switchCamera(cameraSelect) {
  try {
    if (activeWebSocket) {
      activeWebSocket.close();
      activeWebSocket = null;
    }

    await navigator.mediaDevices.getUserMedia({ video: true });
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');

    if (!videoDevices.length) {
      console.error('No se detectaron cámaras');
      return;
    }

    populateCameraSelect(cameraSelect, videoDevices);

    cameraSelect.addEventListener('change', async (event) => {
      await startSelectedCamera(event.target.value);
    });

    await startSelectedCamera(videoDevices[0].deviceId);

  } catch (err) {
    console.error('Error inicializando cámaras:', err);
  }
}

async function startSelectedCamera(deviceId) {
  try {
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

    activeWebSocket = initVideoStream(video);

  } catch (err) {
    console.error('Error al cambiar cámara:', err);
  }
}

function populateCameraSelect(selectElement, devices) {
  selectElement.innerHTML = devices.map((device, index) => `
    <option value="${device.deviceId}">
      ${device.label || `Cámara ${index + 1}`}
    </option>
  `).join('');
}
