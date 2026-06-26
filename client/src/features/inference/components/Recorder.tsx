// Recorder.tsx — boton de grabacion del canvas de salida. Toda la logica (grabar +
// inyectar duracion con fix-webm-duration + descargar) vive en useRecorder.

import { type RefObject } from 'react';
import { Button } from '@/shared/ui/Button';
import { useRecorder } from '../hooks/useRecorder';

export function Recorder({ canvasRef }: { canvasRef: RefObject<HTMLCanvasElement | null> }) {
  const { recording, error, start, stop } = useRecorder(canvasRef);

  return (
    <div className="space-y-1">
      <Button
        variant={recording ? 'danger' : 'primary'}
        className="w-full"
        onClick={recording ? stop : start}
      >
        {recording ? 'Detener grabacion' : 'Iniciar grabacion'}
      </Button>
      {error && <p className="text-xs text-danger">{error}</p>}
    </div>
  );
}
