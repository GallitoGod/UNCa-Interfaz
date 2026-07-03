// models.ts — endpoints REST de modelos para la vista de inferencia.

import { api } from '@/shared/api/axios';
import type { ModelType } from '@/shared/api/types';

// GET /get_models -> { models: string[] } (configs con archivo de pesos).
export async function getModels(): Promise<string[]> {
  const { data } = await api.get<{ models: string[] }>('/get_models');
  return data.models;
}

// POST /select_model { model_name } -> carga + valida en el backend.
export async function selectModel(modelName: string): Promise<void> {
  await api.post('/select_model', { model_name: modelName });
}

// GET /configs/{name} -> model_type real del config (para enrutar la estrategia del
// workspace). null si el modelo no tiene config o no declara tipo.
export async function getModelType(name: string): Promise<ModelType | null> {
  const { data } = await api.get<{ config: { model_type?: ModelType } | null }>(`/configs/${name}`);
  return data.config?.model_type ?? null;
}
