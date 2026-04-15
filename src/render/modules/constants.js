const port = 8000;
const localHost     = `http://127.0.0.1:${port}`;
const websocketBase = `ws://127.0.0.1:${port}`;

export const loadModelUrl     = `${localHost}/get_models`;
export const selectModelUrl   = `${localHost}/select_model`;
export const confidenceUrl    = `${localHost}/config/confidence`;
export const colorsUrl        = `${localHost}/config/colors`;
export const inferenceLogsUrl = `${localHost}/logs/inference`;
export const metricsUrl       = `${localHost}/metrics`;
export const streamUrl        = `${websocketBase}/video_stream`;
