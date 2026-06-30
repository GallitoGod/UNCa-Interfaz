// MetricsPanel.tsx — grilla de metricas del panel derecho (StatTiles). Permanente:
// hace polling mientras la vista de inferencia esta montada. Comparte la query
// ['metrics'] con el overlay del feed (TanStack dedupea por queryKey).

import { useMetrics } from '../hooks/useDiagnostics';
import { StatTile } from '@/shared/ui/StatTile';

export function MetricsPanel() {
  const { data } = useMetrics(true);

  // "--" hasta que el backend tenga datos (modelo cargado + frames procesados).
  const fmt = (n: number | undefined) => (n === undefined ? '--' : n.toFixed(1));
  const fmt0 = (n: number | undefined) => (n === undefined ? '--' : Math.round(n));

  return (
    <div className="grid grid-cols-2 gap-2">
      <StatTile value={fmt0(data?.fps_avg)} label="FPS" />
      <StatTile value={fmt(data?.p95_ms)} unit="ms" label="P95" accent />
      <StatTile value={fmt(data?.inf_avg_ms)} unit="ms" label="Inferencia" />
      <StatTile value={fmt(data?.avg_ms)} unit="ms" label="Total" />
    </div>
  );
}
