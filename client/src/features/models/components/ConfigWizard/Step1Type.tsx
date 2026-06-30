// Step1Type.tsx — eleccion del tipo de modelo. Al cambiar de tipo, el wizard pide
// el template del nuevo tipo y reemplaza output (lo maneja onSelectType en el padre).

import type { ModelConfig, ModelType } from '@/shared/api/types';
import { cn } from '@/shared/ui/cn';

// enabled=false: el pipeline del backend aun no esta implementado (TaskNotImplemented->501).
// Las tarjetas se muestran como "proximamente" pero no son seleccionables.
const TYPES: { value: ModelType; label: string; desc: string; enabled: boolean }[] = [
  { value: 'detection', label: 'Deteccion', desc: 'Localiza objetos con bounding boxes.', enabled: true },
  { value: 'classification', label: 'Clasificacion', desc: 'Asigna una o varias clases a la imagen.', enabled: false },
  { value: 'segmentation', label: 'Segmentacion', desc: 'Asigna una clase a cada pixel.', enabled: false },
];

export function Step1Type({
  config,
  onSelectType,
}: {
  config: ModelConfig;
  onSelectType: (type: ModelType) => void;
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      {TYPES.map((t) => {
        const active = config.model_type === t.value;
        return (
          <button
            key={t.value}
            type="button"
            disabled={!t.enabled}
            aria-disabled={!t.enabled}
            onClick={() => t.enabled && onSelectType(t.value)}
            className={cn(
              'rounded-[var(--radius-md)] border p-4 text-left transition-colors',
              !t.enabled
                ? 'cursor-not-allowed border-border bg-control/40 opacity-40'
                : active
                  ? 'border-accent bg-accent-soft'
                  : 'border-border bg-control hover:border-border-strong',
            )}
          >
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-fg">{t.label}</span>
              {!t.enabled && <span className="lbl">proximamente</span>}
            </div>
            <div className="mt-1 text-xs text-fg-muted">{t.desc}</div>
          </button>
        );
      })}
    </div>
  );
}
