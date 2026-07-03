// diagnostics.ts — endpoints REST de confianza, metricas y logs.

import { api } from '@/shared/api/axios';

// POST /config/confidence { value: 0..1 } -> umbral en vivo.
export async function updateConfidence(value: number): Promise<void> {
  await api.post('/config/confidence', { value });
}

export interface Metrics {
  fps_avg: number;
  inf_avg_ms: number;
  avg_ms: number;
  p95_ms: number;
}

// GET /metrics -> { status, metrics }. Devuelve null si el backend no tiene datos.
export async function getMetrics(): Promise<Metrics | null> {
  const { data } = await api.get<{ status: string; metrics?: Metrics }>('/metrics');
  return data.status === 'ok' && data.metrics ? data.metrics : null;
}

export interface InferenceLog {
  timestamp: string;
  error: string;
}

// GET /logs/inference -> { logs: [{ timestamp, error }] } (ultimos 50).
export async function getInferenceLogs(): Promise<InferenceLog[]> {
  const { data } = await api.get<{ logs: InferenceLog[] }>('/logs/inference');
  return data.logs ?? [];
}
