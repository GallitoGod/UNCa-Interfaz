// LogPanel.tsx — ultimos errores de inferencia. Polling solo cuando esta abierto.
// Render en JSX (elimina el innerHTML del codigo viejo).

import { useInferenceLogs } from '../hooks/useDiagnostics';

export function LogPanel({ open }: { open: boolean }) {
  const { data } = useInferenceLogs(open);
  if (!open) return null;

  return (
    <div className="max-h-40 overflow-y-auto rounded-[var(--radius-md)] border border-border bg-surface p-2">
      {!data || data.length === 0 ? (
        <p className="px-1 py-0.5 text-xs text-fg-subtle">Sin errores registrados.</p>
      ) : (
        <ul className="space-y-1">
          {/* Mas reciente primero. */}
          {data
            .slice()
            .reverse()
            .map((entry, i) => (
              <li key={i} className="text-xs">
                <span className="mr-2 font-mono text-fg-subtle">{entry.timestamp}</span>
                <span className="text-danger">{entry.error}</span>
              </li>
            ))}
        </ul>
      )}
    </div>
  );
}
