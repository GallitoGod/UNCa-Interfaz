const port = 8000;
const localHost = `http://127.0.0.1:${port}`;
const websocketBase = `ws://127.0.0.1:${port}`;

export const loadModelUrl = `${localHost}/get_models`;
export const selectModelUrl = `${localHost}/select_model`;
export const confidenceUrl = `${localHost}/config/confidence`;
export const inferenceLogsUrl = `${localHost}/logs/inference`;
export const metricsUrl = `${localHost}/metrics`;
export const streamUrl = `${websocketBase}/video_stream`;

// Snapshot de metricas de la sesion (Fase 4 tarea 4)
export const metricsSnapshotUrl = `${localHost}/metrics/snapshot`;

// Single source of truth de configs (Fase 3). configTemplateUrl recibe el
// model_type por path; configsUrl recibe el nombre del modelo por path.
export const configTemplateUrl = (modelType) =>
  `${localHost}/config/template/${modelType}`;
export const configsUrl = (name) => `${localHost}/configs/${name}`;

// Forma del tensor crudo de salida (columna izquierda de la vista de mapeo, Fase 4)
export const modelOutputShapeUrl = (name) =>
  `${localHost}/model/output_shape/${name}`;
