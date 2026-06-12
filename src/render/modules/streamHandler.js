import { streamUrl } from './constants.js';
import { drawDetections } from './overlay.js';

// Protocolo: se envia el frame JPEG como binario y el backend responde UN JSON
// {detections: [[x1,y1,x2,y2,conf,cls], ...], error}. El dibujo es local: se
// repinta el mismo frame enviado y se superponen las cajas.
const RESPONSE_TIMEOUT_MS = 3000; // red de seguridad: nunca quedar esperando para siempre

export function initVideoStream(videoElement, { mirror = false } = {}) {
  const outputCanvas = document.getElementById('outputCanvas');
  const outputCtx = outputCanvas.getContext('2d');

  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d');

  let ws;
  let animationFrameId = null;
  let waitingForResponse = false;
  let waitingSince = 0;
  let intentionallyClosed = false;
  let retryDelay = 1000;

  function connect() {
    ws = new WebSocket(streamUrl);

    ws.onopen = () => {
      console.log('WebSocket conectado');
      retryDelay = 1000;
      waitingForResponse = false;
      startFrameLoop();
    };

    ws.onclose = () => {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
      if (!intentionallyClosed) {
        console.log(`WebSocket cerrado. Reconectando en ${retryDelay}ms...`);
        setTimeout(connect, retryDelay);
        retryDelay = Math.min(retryDelay * 2, 10000);
      } else {
        console.log('WebSocket cerrado.');
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };

    ws.onmessage = (event) => {
      waitingForResponse = false;

      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch {
        console.warn('Respuesta del stream no es JSON valido');
        return;
      }

      // Repintar el frame que se envio (sigue intacto en captureCanvas:
      // con un solo frame en vuelo no se sobreescribe hasta la proxima captura)
      if (
        outputCanvas.width !== captureCanvas.width ||
        outputCanvas.height !== captureCanvas.height
      ) {
        outputCanvas.width = captureCanvas.width;
        outputCanvas.height = captureCanvas.height;
      }
      outputCtx.drawImage(captureCanvas, 0, 0);

      if (payload.detections && payload.detections.length > 0) {
        drawDetections(outputCtx, payload.detections);
      }
      // "no_model" es estado normal antes de seleccionar modelo: se muestra el frame solo
      if (payload.error && payload.error !== 'no_model') {
        console.warn(
          'Error de stream:',
          payload.error,
          '— ver /logs/inference'
        );
      }
    };
  }

  function startFrameLoop() {
    function tick() {
      // Anti-deadlock: si el backend no respondio en un tiempo razonable,
      // soltar la espera y seguir mandando frames.
      if (
        waitingForResponse &&
        performance.now() - waitingSince > RESPONSE_TIMEOUT_MS
      ) {
        console.warn(
          'Stream: respuesta demorada, se reanuda el envio de frames'
        );
        waitingForResponse = false;
      }

      if (
        !waitingForResponse &&
        ws.readyState === WebSocket.OPEN &&
        videoElement.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA
      ) {
        const vw = videoElement.videoWidth;
        const vh = videoElement.videoHeight;
        if (vw > 0 && vh > 0) {
          captureCanvas.width = vw;
          captureCanvas.height = vh;
          if (mirror) {
            // Efecto espejo SOLO para camara: se hace aca, en el cliente,
            // asi los archivos subidos no quedan espejados.
            captureCtx.save();
            captureCtx.scale(-1, 1);
            captureCtx.drawImage(videoElement, -vw, 0);
            captureCtx.restore();
          } else {
            captureCtx.drawImage(videoElement, 0, 0);
          }

          waitingForResponse = true;
          waitingSince = performance.now();
          captureCanvas.toBlob(
            (blob) => {
              if (blob && ws.readyState === WebSocket.OPEN) {
                ws.send(blob); // binario: sin overhead de base64
              } else {
                waitingForResponse = false;
              }
            },
            'image/jpeg',
            0.8
          );
        }
      }
      animationFrameId = requestAnimationFrame(tick);
    }
    tick();
  }

  connect();

  return {
    close() {
      intentionallyClosed = true;
      cancelAnimationFrame(animationFrameId);
      if (ws) ws.close();
    },
  };
}
