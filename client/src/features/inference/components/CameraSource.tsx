// CameraSource.tsx — selector de camara. Escribe la fuente en el streamStore;
// el orquestador (useVisionSession) abre el media y el stream.

import { useEffect } from 'react';
import { useCameras } from '../hooks/useCameras';
import { useStreamStore } from '../store/streamStore';
import { Button } from '@/shared/ui/Button';

export function CameraSource() {
  const { cameras, refresh } = useCameras();
  const source = useStreamStore((s) => s.source);
  const setCameraSource = useStreamStore((s) => s.setCameraSource);
  const selected = source.kind === 'camera' ? source.deviceId : '';

  // Auto-seleccionar la primera camara disponible si no hay una elegida.
  useEffect(() => {
    if (cameras.length > 0 && source.kind !== 'camera') {
      setCameraSource(cameras[0].deviceId);
    }
  }, [cameras, source.kind, setCameraSource]);

  // Nombre completo de la camara elegida, para el tooltip: el <select> recorta el
  // texto y el usuario tiene que poder leerlo entero de alguna forma.
  const nombreSeleccionado =
    cameras.find((c) => c.deviceId === selected)?.label ?? '';

  return (
    <div className="space-y-2">
      <label className="lbl">Camara</label>
      <div className="flex min-w-0 gap-2">
        <select
          value={selected}
          onChange={(e) => setCameraSource(e.target.value)}
          title={nombreSeleccionado || undefined}
          // min-w-0 + truncate: sin min-w-0 el <select> NO baja de su ancho
          // intrinseco —el de la opcion mas larga— y una camara con nombre largo
          // (tipico: "HD Pro Webcam C920 (046d:082d)") ensancha toda la columna y
          // le mete una barra de scroll horizontal al panel. Mismo patron que
          // ModelRow, que ya recortaba los nombres de modelo.
          // La lista desplegada la dibuja el sistema operativo y NO se recorta:
          // al elegir se sigue viendo el nombre completo.
          className="h-9 min-w-0 flex-1 truncate rounded-[var(--radius-sm)] border border-border bg-control px-2 text-sm text-fg focus-visible:outline-none focus-visible:border-accent"
        >
          {cameras.length === 0 && <option value="">Sin camaras</option>}
          {cameras.map((c, i) => (
            <option key={c.deviceId} value={c.deviceId}>
              {c.label || `Camara ${i + 1}`}
            </option>
          ))}
        </select>
        <Button variant="outline" size="sm" onClick={() => void refresh()} aria-label="Actualizar camaras">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
            <path d="M21 3v5h-5" />
            <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
            <path d="M8 16H3v5" />
          </svg>
        </Button>
      </div>
    </div>
  );
}
