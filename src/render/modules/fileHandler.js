import { initVideoStream } from './streamHandler.js';
import { stopCurrentStream } from './cameraSwitcher.js';
import { streamUrl } from './constants.js';
import { drawDetections } from './overlay.js';

let activeFileWs = null;

export function handleFileUpload(file) {
  if (file.type.startsWith('video/')) {
    _handleVideo(file);
  } else if (file.type.startsWith('image/')) {
    _handleImage(file);
  } else {
    console.error('Tipo de archivo no soportado:', file.type);
  }
}

function _handleVideo(file) {
  stopCurrentStream();
  if (activeFileWs) {
    activeFileWs.close();
    activeFileWs = null;
  }

  const video = document.getElementById('video');
  const noVideoMsg = document.querySelector('.no-video-message');
  if (noVideoMsg) noVideoMsg.style.display = 'none';

  video.src = URL.createObjectURL(file);
  video.muted = true;
  video.loop = true;
  video.play();

  // sin mirror: los archivos se procesan tal cual (el espejo es solo de camara)
  activeFileWs = initVideoStream(video, { mirror: false });
}

function _handleImage(file) {
  stopCurrentStream();
  if (activeFileWs) {
    activeFileWs.close();
    activeFileWs = null;
  }

  const noVideoMsg = document.querySelector('.no-video-message');
  if (noVideoMsg) noVideoMsg.style.display = 'none';

  const outputCanvas = document.getElementById('outputCanvas');
  const outputCtx = outputCanvas.getContext('2d');

  const img = new Image();
  img.onload = () => {
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = img.width;
    tmpCanvas.height = img.height;
    tmpCanvas.getContext('2d').drawImage(img, 0, 0);

    const ws = new WebSocket(streamUrl);

    ws.onopen = () => {
      tmpCanvas.toBlob(
        (blob) => {
          if (blob) ws.send(blob);
          else ws.close();
        },
        'image/jpeg',
        0.9
      );
    };

    ws.onmessage = (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch {
        ws.close();
        return;
      }
      outputCanvas.width = img.width;
      outputCanvas.height = img.height;
      outputCtx.drawImage(img, 0, 0);
      if (payload.detections && payload.detections.length > 0) {
        drawDetections(outputCtx, payload.detections);
      }
      if (payload.error) {
        console.warn('Error al procesar imagen:', payload.error);
      }
      ws.close();
    };

    ws.onerror = (err) => console.error('WS error al procesar imagen:', err);
  };
  img.src = URL.createObjectURL(file);
}
