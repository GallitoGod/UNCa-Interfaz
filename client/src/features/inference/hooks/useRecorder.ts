// useRecorder.ts — graba el canvas de salida a .webm y lo descarga, inyectando la
// metadata de DURACION con fix-webm-duration. MediaRecorder produce webm sin duracion
// (no se puede hacer seek / muestra duracion infinita); esta es la regla estricta del
// SDD 4.1.3. La logica vive aca (hook) y Recorder.tsx queda como un boton fino.

import { useCallback, useRef, useState, type RefObject } from 'react';
import fixWebmDuration from 'fix-webm-duration';

// captureStream existe en HTMLCanvasElement pero no siempre esta en los tipos DOM.
type CanvasWithCapture = HTMLCanvasElement & { captureStream(fps?: number): MediaStream };

export interface RecorderControls {
  recording: boolean;
  error: string | null;
  start: () => void;
  stop: () => void;
}

export function useRecorder(canvasRef: RefObject<HTMLCanvasElement | null>): RecorderControls {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const startedAt = useRef(0); // performance.now() del inicio, para medir la duracion real

  const download = useCallback((blob: Blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'grabacion.webm';
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  const start = useCallback(() => {
    setError(null);
    const canvas = canvasRef.current as CanvasWithCapture | null;
    if (!canvas) {
      setError('No hay canvas de salida para grabar.');
      return;
    }
    if (typeof MediaRecorder === 'undefined') {
      setError('MediaRecorder no esta disponible en este entorno.');
      return;
    }

    let recorder: MediaRecorder;
    try {
      const stream = canvas.captureStream(30);
      recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    } catch (e) {
      setError(`No se pudo iniciar la grabacion: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }

    chunks.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.current.push(e.data);
    };
    recorder.onstop = () => {
      const raw = new Blob(chunks.current, { type: 'video/webm' });
      chunks.current = [];
      const durationMs = performance.now() - startedAt.current;
      // Inyecta la duracion; si fix-webm-duration falla, descargamos el blob crudo
      // igual (mejor un webm sin metadata que perder la grabacion).
      fixWebmDuration(raw, durationMs, { logger: false })
        .then(download)
        .catch((e) => {
          console.warn('fix-webm-duration fallo, se descarga sin metadata:', e);
          download(raw);
        });
    };

    startedAt.current = performance.now();
    recorder.start();
    recorderRef.current = recorder;
    setRecording(true);
  }, [canvasRef, download]);

  const stop = useCallback(() => {
    recorderRef.current?.stop(); // dispara onstop -> fix + descarga
    recorderRef.current = null;
    setRecording(false);
  }, []);

  return { recording, error, start, stop };
}
