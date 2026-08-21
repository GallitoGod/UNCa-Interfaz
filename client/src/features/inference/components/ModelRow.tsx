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
  loading?: boolean; // este modelo se esta cargando en el backend
  disabled?: boolean; // hay otra carga en curso
}

export function ModelRow({
  name,
  active,
  onSelect,
  format,
  loading = false,
  disabled = false,
}: ModelRowProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={active}
      aria-busy={loading}
      disabled={disabled || loading}
      className={cn(
        'flex w-full items-center gap-2 rounded-[8px] border px-3 py-2.5 text-left',
        'transition-colors duration-150 focus-visible:outline-none active:scale-[0.99]',
        'disabled:cursor-default',
        active
          ? 'border-accent bg-accent-soft'
          : 'border-border bg-control hover:border-border-strong',
        // Cargando: se resalta el que esta ocupado; el resto se atenua.
        loading && 'border-accent/60',
        disabled && 'opacity-40',
      )}
    >
      {/* Punto de estado: con glow cuando esta activo, latiendo mientras carga. */}
      <span
        className={cn(
          'size-[7px] shrink-0 rounded-full',
          loading
            ? 'animate-pulse bg-accent shadow-[0_0_8px_#34d6ff]'
            : active
              ? 'bg-accent shadow-[0_0_8px_#34d6ff]'
              : 'bg-[#3a4658]',
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
      {loading ? (
        // Reemplaza al badge de formato mientras dura la carga: el usuario necesita
        // saber que el sistema esta trabajando, no en que formato esta el archivo.
        <span className="shrink-0 animate-pulse font-mono text-[9px] font-semibold uppercase tracking-[0.08em] text-accent">
          cargando
        </span>
      ) : (
        format && (
          <span
            className={cn(
              'rounded-[4px] px-1.5 py-0.5 font-mono text-[9px] font-semibold',
              active ? 'bg-accent-soft text-accent' : 'bg-white/5 text-label',
            )}
          >
            {format}
          </span>
        )
      )}
    </button>
  );
}
