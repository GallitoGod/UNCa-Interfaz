// ModelCard.tsx — bloque "lego" de un archivo de pesos (spec Modelos §03): studs en
// relieve sobre el borde superior, badge de formato (extension), nombre y estado de
// config abajo. Seleccionado: borde + glow cian, studs y badge encendidos.

import type { ModelEntry } from '../api/models';
import { cn } from '@/shared/ui/cn';

export function ModelCard({
  entry,
  selected,
  onSelect,
}: {
  entry: ModelEntry;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        'relative flex min-h-[110px] flex-col rounded-[var(--radius-md)] border px-[13px] pb-[13px] pt-3.5 text-left transition-colors',
        selected
          ? 'border-accent-border bg-accent-soft shadow-[0_0_0_1px_var(--c-accent-border),0_0_26px_-8px_var(--c-accent)]'
          : 'border-border bg-control hover:border-border-strong',
      )}
    >
      {/* Studs: tres tetones tipo lego que sobresalen del borde superior. Puro
          caracter visual (cero data); se encienden en cian al seleccionar. */}
      <span aria-hidden="true" className="absolute -top-[5px] left-[15px] flex gap-[5px]">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className={cn('h-1.5 w-3 rounded-t-[3px]', selected ? 'bg-accent' : 'bg-border-strong')}
          />
        ))}
      </span>

      {/* Badge de formato derivado de la extension (ONNX / TFLITE / PT / ...). */}
      <span
        className={cn(
          'self-start rounded-[5px] border px-1.5 py-[3px] font-mono text-[9.5px] font-semibold uppercase tracking-[0.5px]',
          selected ? 'border-accent-border text-accent' : 'border-border bg-white/5 text-fg-muted',
        )}
      >
        {entry.ext}
      </span>

      <span
        className="mt-[13px] truncate text-[14.5px] font-semibold tracking-[-0.01em] text-fg"
        title={entry.file}
      >
        {entry.baseName}
      </span>

      {/* Estado de config anclado abajo; el punto del activo con config lleva glow. */}
      <span
        className={cn(
          'mt-auto flex items-center gap-1.5 pt-[11px] font-mono text-[11px]',
          entry.hasConfig ? 'text-success' : 'text-fg-subtle',
        )}
      >
        <span
          className={cn(
            'size-[7px] shrink-0 rounded-full',
            entry.hasConfig ? 'bg-success' : 'bg-label',
            selected && entry.hasConfig && 'shadow-[0_0_8px_1px_currentColor]',
          )}
        />
        {entry.hasConfig ? 'config' : 'sin config'}
      </span>
    </button>
  );
}
