// configs.ts — endpoints REST de templates y guardado de config (backend FastAPI).
// El guardado valida contra ModelConfig estricto en el backend antes de escribir.

import { api } from '@/shared/api/axios';
import type { AnchorConfig, ModelConfig, ModelType } from '@/shared/api/types';

export interface ConfigTemplateResponse {
  config: ModelConfig;
  anchor_defaults: AnchorConfig | null;
}

// GET /config/template/{type} -> defaults desde Pydantic (single source of truth).
export async function getConfigTemplate(type: ModelType): Promise<ConfigTemplateResponse> {
  const { data } = await api.get<ConfigTemplateResponse>(`/config/template/${type}`);
  return data;
}

// POST /configs/{name} -> valida + escribe configs/<name>.json.
export async function saveConfig(name: string, config: ModelConfig): Promise<void> {
  await api.post(`/configs/${name}`, config);
}

// GET /configs/{name} -> config existente, o null si el modelo no tiene una todavia.
// Reemplaza el viejo IPC 'configs:read'. Un JSON corrupto en el backend responde 500
// y el ApiError sube al hook (el wizard cae al template — regla SDD 4.1.4: no bloquea).
export async function getConfig(name: string): Promise<ModelConfig | null> {
  const { data } = await api.get<{ config: ModelConfig | null }>(`/configs/${name}`);
  return data.config;
}
