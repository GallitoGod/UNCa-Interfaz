// UnsupportedOverlay.tsx — aviso cuando el modelo activo es de un tipo cuya
// visualizacion todavia no esta implementada (CLS/SEG stubs). Sin throw: el frame
// se sigue mostrando, solo no hay capa de resultados.

import type { ModelType } from '@/shared/api/types';

const LABEL: Record<ModelType, string> = {
  detection: 'Deteccion',
  classification: 'Clasificacion',
  segmentation: 'Segmentacion',
};

export function UnsupportedOverlay({ type }: { type: ModelType }) {
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-center p-3">
      <div className="rounded-[var(--radius-md)] border border-warn/40 bg-canvas/80 px-3 py-1.5 text-xs text-warn backdrop-blur-sm">
        {LABEL[type]}: la visualizacion aun no esta implementada
      </div>
    </div>
  );
}
