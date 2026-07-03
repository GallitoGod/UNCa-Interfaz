// MetricsHUD.tsx — overlay compacto sobre el feed: lectura de instrumento al vuelo
// (solo FPS). El desglose completo (inf/total/P95) vive en el MetricsPanel de la
// derecha; aca no se duplica para no competir por foco. Comparte la query ['metrics'].

import { useMetrics } from '../hooks/useDiagnostics';

export function MetricsHUD({ open }: { open: boolean }) {
  const { data } = useMetrics(open);
  if (!open) return null;

  const fps = data?.fps_avg === undefined ? '--' : Math.round(data.fps_avg);

  return (
    <div className="pointer-events-none absolute left-3 top-3 flex items-baseline gap-1.5 rounded-[var(--radius-md)] border border-border bg-[#05080c]/80 px-3 py-1.5 font-mono backdrop-blur-sm [font-variant-numeric:tabular-nums]">
      <span className="text-base font-bold text-fg">{fps}</span>
      <span className="text-[10px] font-semibold uppercase tracking-[0.5px] text-label">
        FPS
      </span>
    </div>
  );
}
