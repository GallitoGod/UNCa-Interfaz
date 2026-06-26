// MetricsHUD.tsx — overlay de metricas sobre el video (monospace). Hace polling
// solo cuando esta abierto (useMetrics(open)).

import { useMetrics } from '../hooks/useDiagnostics';

export function MetricsHUD({ open }: { open: boolean }) {
  const { data } = useMetrics(open);
  if (!open) return null;

  const fmt = (n: number | undefined) => (n === undefined ? '--' : n.toFixed(1));

  return (
    <div className="pointer-events-none absolute left-3 top-3 space-y-0.5 rounded-[var(--radius-md)] border border-border bg-canvas/70 px-3 py-2 font-mono text-xs text-fg backdrop-blur-sm">
      <div>FPS:   {fmt(data?.fps_avg)}</div>
      <div>Inf:   {fmt(data?.inf_avg_ms)} ms</div>
      <div>Total: {fmt(data?.avg_ms)} ms</div>
      <div>P95:   {fmt(data?.p95_ms)} ms</div>
    </div>
  );
}
