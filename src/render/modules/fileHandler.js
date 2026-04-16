import { initVideoStream } from './streamHandler.js';
import { stopCurrentStream } from './cameraSwitcher.js';
import { streamUrl } from './constants.js';

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
  if (activeFileWs) { activeFileWs.close(); activeFileWs = null; }

  const video = document.getElementById('video');
  const noVideoMsg = document.querySelector('.no-video-message');
  if (noVideoMsg) noVideoMsg.style.display = 'none';

  video.src = URL.createObjectURL(file);
  video.muted = true;
  video.loop = true;
  video.play();

  activeFileWs = initVideoStream(video);
}

function _handleImage(file) {
  stopCurrentStream();
  if (activeFileWs) { activeFileWs.close(); activeFileWs = null; }

  const noVideoMsg = document.querySelector('.no-video-message');
  if (noVideoMsg) noVideoMsg.style.display = 'none';

  const outputCanvas = document.getElementById('outputCanvas');
  const outputCtx    = outputCanvas.getContext('2d');

  const img = new Image();
  img.onload = () => {
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width  = img.width;
    tmpCanvas.height = img.height;
    tmpCanvas.getContext('2d').drawImage(img, 0, 0);

    const ws = new WebSocket(streamUrl);

    ws.onopen = () => {
      ws.send(tmpCanvas.toDataURL('image/jpeg', 0.8));
    };

    ws.onmessage = (event) => {
      const annotated = new Image();
      annotated.onload = () => {
        outputCanvas.width  = annotated.width;
        outputCanvas.height = annotated.height;
        outputCtx.drawImage(annotated, 0, 0);
      };
      annotated.src = 'data:image/jpeg;base64,' + event.data;
      ws.close();
    };

    ws.onerror = (err) => console.error('WS error al procesar imagen:', err);
  };
  img.src = URL.createObjectURL(file);
}
