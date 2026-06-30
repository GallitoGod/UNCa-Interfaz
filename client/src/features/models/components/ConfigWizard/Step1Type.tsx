// Step1Type.tsx — eleccion del tipo de modelo. Al cambiar de tipo, el wizard pide
// el template del nuevo tipo y reemplaza output (lo maneja onSelectType en el padre).

import type { ModelConfig, ModelType } from '@/shared/api/types';
import { cn } from '@/shared/ui/cn';

const TYPES: { value: ModelType; label: string; desc: string }[] = [
  { value: 'detection', label: 'Deteccion', desc: 'Localiza objetos con bounding boxes.' },
  { value: 'classification', label: 'Clasificacion', desc: 'Asigna una o varias clases a la imagen.' },
  { value: 'segmentation', label: 'Segmentacion', desc: 'Asigna una clase a cada pixel.' },
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
            onClick={() => onSelectType(t.value)}
            className={cn(
              'rounded-[var(--radius-md)] border p-4 text-left transition-colors',
              active
                ? 'border-accent bg-accent-soft'
                : 'border-border bg-control hover:border-border-strong',
            )}
          >
            <div className="text-sm font-medium text-fg">{t.label}</div>
            <div className="mt-1 text-xs text-fg-muted">{t.desc}</div>
          </button>
        );
      })}
    </div>
  );
}
