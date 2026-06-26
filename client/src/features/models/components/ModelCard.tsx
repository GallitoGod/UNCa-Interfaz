// ModelCard.tsx — tarjeta de un archivo de pesos con su estado de config.

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
        'rounded-[var(--radius-md)] border p-3 text-left transition-colors',
        selected ? 'border-accent bg-accent/5' : 'border-border hover:border-border-strong',
      )}
    >
      <div className="flex items-center justify-between">
        <span className="rounded bg-control px-1.5 py-0.5 font-mono text-[10px] uppercase text-fg-muted">
          {entry.ext}
        </span>
        {entry.hasConfig ? (
          <span className="text-xs text-success">● config</span>
        ) : (
          <span className="text-xs text-fg-subtle">○ sin config</span>
        )}
      </div>
      <div className="mt-2 truncate text-sm font-medium text-fg" title={entry.file}>
        {entry.baseName}
      </div>
    </button>
  );
}
