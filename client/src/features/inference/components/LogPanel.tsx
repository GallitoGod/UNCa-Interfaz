// LogPanel.tsx — ultimos errores de inferencia. Polling solo cuando esta abierto.
// Render en JSX (elimina el innerHTML del codigo viejo).

import { useInferenceLogs } from '../hooks/useDiagnostics';

export function LogPanel({ open }: { open: boolean }) {
  const { data } = useInferenceLogs(open);
  if (!open) return null;

  // Sin errores: fila verde "pipeline estable" (estado de sistema, no decoracion).
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center gap-2 rounded-[8px] border border-border bg-control px-3 py-2.5">
        <span className="size-[7px] shrink-0 rounded-full bg-success shadow-[0_0_7px_#28c840]" />
        <span className="text-[11px] text-fg-muted">Sin errores · pipeline estable</span>
      </div>
    );
  }

  return (
    <ul className="max-h-48 space-y-1.5 overflow-y-auto">
      {/* Mas reciente primero. Todas las entradas son errores de inferencia -> punto rojo. */}
      {data
        .slice()
        .reverse()
        .map((entry, i) => (
          <li
            key={i}
            className="flex gap-2.5 rounded-[8px] border border-border bg-control px-3 py-2.5"
          >
            <span className="mt-1 size-[7px] shrink-0 rounded-full bg-danger" />
            <div className="min-w-0 flex-1">
              <div className="mb-1 font-mono text-[9px] text-label">{entry.timestamp}</div>
              <div className="text-[11px] leading-snug text-fg-muted">{entry.error}</div>
            </div>
          </li>
        ))}
    </ul>
  );
}
