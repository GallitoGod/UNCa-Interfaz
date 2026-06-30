// StatTile.tsx — tarjeta de metrica: numero mono grande (heroe del dato) + label
// mono chico debajo. `accent` pinta el numero en cian para destacar un dato (P95).
// Numeros con tabular-nums para que no salten al cambiar de valor (hot path de
// metricas en vivo).

import { cn } from './cn';

interface StatTileProps {
  value: string | number; // ya formateado por el consumidor
  label: string;
  unit?: string; // sufijo chico (ej. "ms")
  accent?: boolean; // dato destacado -> numero en cian
}

export function StatTile({ value, label, unit, accent = false }: StatTileProps) {
  return (
    <div className="rounded-[8px] border border-border bg-control px-3 py-2.5">
      <div
        className={cn(
          'font-mono text-[22px] font-bold leading-none [font-variant-numeric:tabular-nums]',
          accent ? 'text-accent' : 'text-fg',
        )}
      >
        {value}
        {unit && <span className="ml-0.5 text-[10px] text-fg-subtle">{unit}</span>}
      </div>
      <div className="mt-1.5 font-mono text-[9px] font-semibold uppercase tracking-[0.5px] text-label">
        {label}
      </div>
    </div>
  );
}
