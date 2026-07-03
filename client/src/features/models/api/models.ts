// models.ts — endpoints REST de gestion de pesos (listar + subir).
// Reemplaza el viejo IPC de disco (models:list / models:import): ahora TODO pasa por
// el backend HTTP (thin client sin acceso a disco — ver SDD). Como efecto colateral,
// la vista de Modelos ya funciona tambien en un browser de dev (sin Electron).

import { api } from '@/shared/api/axios';

// Una entrada de la lista: un archivo de pesos en models/ con su estado de config.
// (Antes vivia en shared/electron/uncaApi.ts, ya eliminado.)
export interface ModelEntry {
  file: string; // nombre del archivo (ej: "yolov7-tiny.onnx")
  ext: string; // extension sin punto (onnx, tflite, pt, ...)
  baseName: string; // nombre sin extension (clave de la config)
  hasConfig: boolean; // existe configs/<baseName>.json
}

// GET /models -> todos los pesos de models/ con su flag hasConfig (vista Modelos).
export async function listModels(): Promise<ModelEntry[]> {
  const { data } = await api.get<{ models: ModelEntry[] }>('/models');
  return data.models;
}

// POST /models/upload -> sube UN archivo por multipart. onProgress recibe la fraccion
// subida en [0,1] (solo si el browser reporta el total). El backend valida extension y
// nombre seguro ANTES de leer el stream, asi un archivo invalido falla rapido.
export async function uploadModel(
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<{ ok: boolean; file: string }> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post<{ ok: boolean; file: string }>('/models/upload', form, {
    // Axios agrega el boundary correcto al detectar FormData con este content-type.
    headers: { 'Content-Type': 'multipart/form-data' },
    // Pesos grandes: anulamos el timeout por defecto (10s) de la instancia.
    timeout: 0,
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(e.loaded / e.total);
    },
  });
  return data;
}
