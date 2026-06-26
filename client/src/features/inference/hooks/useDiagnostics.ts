// useDiagnostics.ts — hooks de confianza (mutation), metricas y logs (queries con
// polling solo cuando el panel correspondiente esta abierto).

import { useMutation, useQuery } from '@tanstack/react-query';
import { getInferenceLogs, getMetrics, updateConfidence } from '../api/diagnostics';

export function useUpdateConfidence() {
  return useMutation({ mutationFn: updateConfidence });
}

// enabled = HUD abierto; refetch cada 1s mientras tanto (reemplaza el setInterval).
export function useMetrics(enabled: boolean) {
  return useQuery({
    queryKey: ['metrics'],
    queryFn: getMetrics,
    enabled,
    refetchInterval: enabled ? 1000 : false,
  });
}

// enabled = panel de logs abierto; refetch cada 5s.
export function useInferenceLogs(enabled: boolean) {
  return useQuery({
    queryKey: ['inference-logs'],
    queryFn: getInferenceLogs,
    enabled,
    refetchInterval: enabled ? 5000 : false,
  });
}
