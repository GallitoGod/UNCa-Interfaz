// VisionWorkspace.tsx — superficie de presentacion (presentacional).
// Monta el <video> oculto (fuente), el canvas de salida y el overlayRoot (capa HTML).
// La orquestacion del stream vive en useVisionSession (feature inference), que recibe
// estos refs. children = slots superpuestos (ej. MetricsHUD).

import { type ReactNode, type RefObject } from 'react';
import { useWorkspaceStore } from '../store/workspaceStore';
import { getStrategy } from '../services/registry';
import { UnsupportedOverlay } from './UnsupportedOverlay';

interface VisionWorkspaceProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  overlayRef: RefObject<HTMLDivElement | null>;
  hasSource: boolean; // hay una fuente activa (camara/archivo)
  children?: ReactNode; // overlays (HUD)
}

export function VisionWorkspace({
  videoRef,
  canvasRef,
  overlayRef,
  hasSource,
  children,
}: VisionWorkspaceProps) {
  const activeModel = useWorkspaceStore((s) => s.activeModel);
  const unsupported = activeModel ? !getStrategy(activeModel.type).implemented : false;

  return (
    <div className="relative grid h-full place-items-center overflow-hidden rounded-[var(--radius-lg)] border border-border bg-black">
      {/* Fuente: oculta, solo alimenta al stream. */}
      <video ref={videoRef} className="hidden" playsInline muted />

      {/* Salida: el frame es el heroe; object-contain conserva el aspecto. */}
      <canvas
        ref={canvasRef}
        className="max-h-full max-w-full object-contain"
      />

      {/* Capa HTML de overlays (badges de clasificacion, leyendas). */}
      <div ref={overlayRef} className="pointer-events-none absolute inset-0" />

      {/* Estado vacio. */}
      {!hasSource && (
        <div className="absolute inset-0 grid place-items-center">
          <p className="text-sm text-fg-subtle">No hay fuente de video seleccionada</p>
        </div>
      )}

      {unsupported && activeModel && <UnsupportedOverlay type={activeModel.type} />}

      {children}
    </div>
  );
}
