// ModelRow.tsx — fila de la lista de modelos del panel de inferencia.
// Activa: fondo cian-soft + borde cian + punto con glow. Inactiva: tile + punto
// apagado. El badge de formato es opcional (el endpoint /get_models solo da nombres;
// se muestra si el consumidor lo provee).

import { cn } from '@/shared/ui/cn';

interface ModelRowProps {
  name: string;
  active: boolean;
  onSelect: () => void;
  format?: string; // ej. "ONNX" / "TFLite"; opcional
}

export function ModelRow({ name, active, onSelect, format }: ModelRowProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={active}
      className={cn(
        'flex w-full items-center gap-2 rounded-[8px] border px-3 py-2.5 text-left',
        'transition-colors duration-150 focus-visible:outline-none active:scale-[0.99]',
        active
          ? 'border-accent bg-accent-soft'
          : 'border-border bg-control hover:border-border-strong',
      )}
    >
      {/* Punto de estado: con glow cuando esta activo. */}
      <span
        className={cn(
          'size-[7px] shrink-0 rounded-full',
          active ? 'bg-accent shadow-[0_0_8px_#34d6ff]' : 'bg-[#3a4658]',
        )}
      />
      <span
        className={cn(
          'flex-1 truncate font-mono text-xs font-semibold',
          active ? 'text-fg' : 'text-fg-subtle',
        )}
        title={name}
      >
        {name}
      </span>
      {format && (
        <span
          className={cn(
            'rounded-[4px] px-1.5 py-0.5 font-mono text-[9px] font-semibold',
            active ? 'bg-accent-soft text-accent' : 'bg-white/5 text-label',
          )}
        >
          {format}
        </span>
      )}
    </button>
  );
}
