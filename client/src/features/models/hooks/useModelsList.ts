// useModelsList.ts — hooks de gestion de modelos. Todo por REST (TanStack Query):
// el frontend ya no usa IPC de disco (thin client sin disco — ver SDD).

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type { ModelType } from '@/shared/api/types';
import type { ModelConfig } from '@/shared/api/types';
import { ApiError } from '@/shared/api/errors';
import { listModels, uploadModel } from '../api/models';
import { getConfig, getConfigTemplate, saveConfig } from '../api/configs';
import { toBackendConfig } from '../lib/toBackendConfig';

// Lista de archivos de pesos en models/ con su estado de config (GET /models).
export function useModelsList() {
  return useQuery({
    queryKey: ['models-list'],
    queryFn: listModels,
  });
}

// Config existente de un modelo (GET /configs/{name}). null si no tiene.
export function useModelConfig(baseName: string | null) {
  return useQuery({
    queryKey: ['model-config', baseName],
    queryFn: () => getConfig(baseName as string),
    enabled: !!baseName,
  });
}

// Progreso de subida de UN archivo (fraction en [0,1]).
export interface ImportProgress {
  file: string;
  fraction: number;
}

// Resultado del import: que se subio y que fallo, archivo por archivo.
export interface ImportResult {
  uploaded: string[];
  failed: { file: string; error: string }[];
}

// Importar modelos: sube los archivos UNO A UNO (los pesos son grandes; evita saturar
// red/RAM del backend) y acumula exito/error por archivo. La mutacion NO tira: un fallo
// puntual queda en `failed` y no aborta el resto. Invalida las listas al terminar.
export function useImportModels() {
  const qc = useQueryClient();
  return useMutation<ImportResult, ApiError, { files: File[]; onProgress?: (p: ImportProgress) => void }>({
    mutationFn: async ({ files, onProgress }) => {
      const uploaded: string[] = [];
      const failed: { file: string; error: string }[] = [];
      for (const f of files) {
        try {
          const res = await uploadModel(f, (fraction) => onProgress?.({ file: f.name, fraction }));
          uploaded.push(res.file);
        } catch (err) {
          // El interceptor de Axios ya normaliza a ApiError (mensaje legible).
          failed.push({
            file: f.name,
            error: err instanceof ApiError ? err.message : 'error desconocido',
          });
        }
      }
      return { uploaded, failed };
    },
    onSuccess: (res) => {
      if (res.uploaded.length === 0) return; // nada cambio en disco
      void qc.invalidateQueries({ queryKey: ['models-list'] }); // vista Modelos
      void qc.invalidateQueries({ queryKey: ['models'] }); // selector de inferencia (/get_models)
    },
  });
}

// Defaults del backend para un tipo (single source of truth).
export function useConfigTemplate(type: ModelType) {
  return useQuery({
    queryKey: ['config-template', type],
    queryFn: () => getConfigTemplate(type),
    staleTime: Infinity, // los defaults no cambian en runtime
  });
}

// Guardar config: aplica toBackendConfig y postea. Invalida listas al terminar.
export function useSaveConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, config }: { name: string; config: ModelConfig }) =>
      saveConfig(name, toBackendConfig(config)),
    onSuccess: (_data, { name }) => {
      void qc.invalidateQueries({ queryKey: ['models-list'] });
      void qc.invalidateQueries({ queryKey: ['model-config', name] });
      void qc.invalidateQueries({ queryKey: ['models'] }); // /get_models (header)
    },
  });
}
