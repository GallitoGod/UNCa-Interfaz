// videoStream.ts — transporte WS de inferencia (framework-agnostico, sin React).
// Porte fiel del streamHandler.js viejo. Conserva sus invariantes criticas:
//   - 1 frame en vuelo (no se envia el proximo hasta recibir respuesta)
//   - anti-deadlock (timeout de 3s libera la espera)
//   - el frame enviado queda intacto en captureCanvas para repintarlo
//   - reconexion con backoff exponencial (salvo cierre intencional)
//   - mirror aplicado en la captura, solo cuando lo pide la camara

import { STREAM_URL } from '@/shared/api/ws';

export type StreamStatus = 'connecting' | 'open' | 'closed' | 'waiting';

/**
 * Lo que devuelve el backend por cada frame, en UNA de dos formas (paso 3 del plan
 * del 2026-08-21, implementado el 2026-08-26):
 *
 *   'frame' -> BINARIO: el JPEG ya compuesto por el backend (deteccion/segmentacion).
 *              El cliente solo lo pinta: no parsea, no dibuja, no sabe de cajas.
 *   'json'  -> TEXTO: el envelope {task, result, error} de siempre. Lo usan
 *              clasificacion (su resultado es texto, no geometria) y TODOS los
 *              errores, de cualquier tarea.
 *
 * Se discrimina por el TIPO del dato recibido, no por un campo del mensaje.
 */
export type StreamPayload =
  | { kind: 'json'; envelope: unknown }
  | { kind: 'frame'; bitmap: ImageBitmap };

// Decodifica el frame compuesto. Devuelve null si el blob no es una imagen valida:
// un frame roto no debe matar el loop.
async function decodeFrame(data: Blob | ArrayBuffer): Promise<ImageBitmap | null> {
  const blob = data instanceof Blob ? data : new Blob([data], { type: 'image/jpeg' });
  try {
    return await createImageBitmap(blob);
  } catch (e) {
    console.warn('No se pudo decodificar el frame compuesto del backend:', e);
    return null;
  }
}

export interface VideoStreamHandle {
  // Pausa el envio de frames y el <video> SIN cerrar el WS ni soltar la camara
  // (para navegar a otra vista y volver sin reconectar ni repedir permisos).
  pause(): void;
  // Reanuda tras pause(): vuelve a reproducir y reinicia el loop si el WS sigue abierto.
  resume(): void;
  close(): void;
}

export interface VideoStreamOptions {
  videoElement: HTMLVideoElement;
  mirror?: boolean;
  onMessage: (payload: StreamPayload, captureCanvas: HTMLCanvasElement) => void;
  onStatus?: (status: StreamStatus) => void;
}

const RESPONSE_TIMEOUT_MS = 3000; // red de seguridad: nunca esperar para siempre

export function startVideoStream(opts: VideoStreamOptions): VideoStreamHandle {
  const { videoElement, mirror = false, onMessage, onStatus } = opts;

  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d');

  let ws: WebSocket | null = null;
  let animationFrameId: number | null = null;
  let waitingForResponse = false;
  let waitingSince = 0;
  let intentionallyClosed = false;
  let paused = false; // navegacion fuera de Inferencia: loop detenido, WS vivo
  let retryDelay = 1000;
  let lastResponseSeq = 0; // ordena los decodes asincronos de frames compuestos

  function connect() {
    onStatus?.('connecting');
    ws = new WebSocket(STREAM_URL);

    ws.onopen = () => {
      retryDelay = 1000;
      waitingForResponse = false;
      onStatus?.('open');
      // Si reconectamos estando en pausa (navegacion fuera de Inferencia), no
      // arrancamos el loop: lo hara resume() al volver.
      if (!paused) startFrameLoop();
    };

    ws.onclose = () => {
      if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
      if (!intentionallyClosed) {
        onStatus?.('connecting');
        setTimeout(connect, retryDelay);
        retryDelay = Math.min(retryDelay * 2, 10000);
      } else {
        onStatus?.('closed');
      }
    };

    ws.onerror = (err) => console.error('WebSocket error:', err);

    ws.onmessage = (event) => {
      waitingForResponse = false;

      // TEXTO: envelope JSON (clasificacion o error).
      if (typeof event.data === 'string') {
        let envelope: unknown;
        try {
          envelope = JSON.parse(event.data);
        } catch {
          console.warn('Respuesta de texto del stream no es JSON valido');
          return;
        }
        // captureCanvas sigue con el frame que se envio (1 en vuelo): el consumidor
        // lo repinta y superpone su capa.
        onMessage({ kind: 'json', envelope }, captureCanvas);
        return;
      }

      // BINARIO: frame ya compuesto por el backend. El decode es ASINCRONO, asi que
      // se numera la respuesta: si mientras decodificabamos llego una mas nueva,
      // este bitmap se descarta en vez de pintar un frame viejo encima del actual.
      const seq = ++lastResponseSeq;
      void decodeFrame(event.data as Blob).then((bitmap) => {
        if (!bitmap) return;
        if (seq !== lastResponseSeq) {
          bitmap.close(); // llego uno mas nuevo: soltar la memoria del viejo
          return;
        }
        onMessage({ kind: 'frame', bitmap }, captureCanvas);
      });
    };
  }

  function startFrameLoop() {
    function tick() {
      // Anti-deadlock: si el backend no respondio a tiempo, soltar la espera.
      if (waitingForResponse && performance.now() - waitingSince > RESPONSE_TIMEOUT_MS) {
        console.warn('Stream: respuesta demorada, se reanuda el envio');
        waitingForResponse = false;
      }

      if (
        !waitingForResponse &&
        ws?.readyState === WebSocket.OPEN &&
        videoElement.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA
      ) {
        const vw = videoElement.videoWidth;
        const vh = videoElement.videoHeight;
        if (vw > 0 && vh > 0 && captureCtx) {
          captureCanvas.width = vw;
          captureCanvas.height = vh;
          if (mirror) {
            // Espejo SOLO camara: aca, en el cliente, asi los archivos no se espejan.
            captureCtx.save();
            captureCtx.scale(-1, 1);
            captureCtx.drawImage(videoElement, -vw, 0);
            captureCtx.restore();
          } else {
            captureCtx.drawImage(videoElement, 0, 0);
          }

          waitingForResponse = true;
          waitingSince = performance.now();
          onStatus?.('waiting');
          captureCanvas.toBlob(
            (blob) => {
              if (blob && ws?.readyState === WebSocket.OPEN) {
                ws.send(blob); // binario: sin overhead de base64
              } else {
                waitingForResponse = false;
              }
            },
            'image/jpeg',
            0.8,
          );
        }
      }
      animationFrameId = requestAnimationFrame(tick);
    }
    tick();
  }

  connect();

  return {
    pause() {
      if (paused) return;
      paused = true;
      if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
      waitingForResponse = false; // soltar cualquier frame en vuelo
      videoElement.pause();
    },
    resume() {
      if (!paused) return;
      paused = false;
      // play() puede rechazar (autoplay); no es fatal para reanudar el loop.
      void videoElement.play().catch(() => {});
      // Si el WS sigue abierto reiniciamos el loop; si se cayo, el backoff lo
      // reconectara y onopen lo arrancara (paused ya es false).
      if (ws?.readyState === WebSocket.OPEN && animationFrameId === null) {
        startFrameLoop();
      }
    },
    close() {
      intentionallyClosed = true;
      if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
      ws?.close();
    },
  };
}

// Envio one-shot para imagenes: abre un WS efimero, manda un frame y cierra.
export function sendSingleFrame(
  sourceCanvas: HTMLCanvasElement,
  onResult: (payload: StreamPayload, sourceCanvas: HTMLCanvasElement) => void,
): void {
  const ws = new WebSocket(STREAM_URL);

  ws.onopen = () => {
    sourceCanvas.toBlob(
      (blob) => {
        if (blob) ws.send(blob);
        else ws.close();
      },
      'image/jpeg',
      0.9,
    );
  };

  // Mismas dos formas que el stream continuo: texto (envelope) o binario (frame ya
  // compuesto). Aca hay un solo frame en juego, asi que no hace falta ordenar nada.
  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      let envelope: unknown;
      try {
        envelope = JSON.parse(event.data);
      } catch {
        ws.close();
        return;
      }
      onResult({ kind: 'json', envelope }, sourceCanvas);
      ws.close();
      return;
    }

    void decodeFrame(event.data as Blob).then((bitmap) => {
      if (bitmap) onResult({ kind: 'frame', bitmap }, sourceCanvas);
      ws.close();
    });
  };

  ws.onerror = (err) => console.error('WS error al procesar imagen:', err);
}
