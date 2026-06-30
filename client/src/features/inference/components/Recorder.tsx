// Recorder.tsx — boton de grabacion del canvas de salida. Toda la logica (grabar +
// inyectar duracion con fix-webm-duration + descargar) vive en useRecorder.

import { type RefObject } from 'react';
import { cn } from '@/shared/ui/cn';
import { Badge } from '@/shared/ui/Badge';
import { useRecorder } from '../hooks/useRecorder';

// Boton de grabacion estilo transporte: circulo con contorno; el indicador rojo pasa
// de circulo (listo para grabar) a cuadrado (grabando, click para detener).
export function Recorder({ canvasRef }: { canvasRef: RefObject<HTMLCanvasElement | null> }) {
  const { recording, error, start, stop } = useRecorder(canvasRef);

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        onClick={recording ? stop : start}
        aria-label={recording ? 'Detener grabacion' : 'Iniciar grabacion'}
        aria-pressed={recording}
        className={cn(
          'grid size-9 shrink-0 place-items-center rounded-full border bg-control',
          'transition-colors duration-150 focus-visible:outline-none active:scale-95',
          recording
            ? 'border-[rgba(255,77,79,0.35)]'
            : 'border-border hover:border-border-strong',
        )}
      >
        <span
          className={cn(
            'size-3 bg-danger transition-all duration-150',
            recording ? 'rounded-[2px]' : 'rounded-full',
          )}
        />
      </button>

      {recording ? (
        <Badge variant="rec">REC</Badge>
      ) : (
        <span className="text-[11px] text-fg-subtle">Grabar salida</span>
      )}

      {error && <p className="text-xs text-danger">{error}</p>}
    </div>
  );
}
