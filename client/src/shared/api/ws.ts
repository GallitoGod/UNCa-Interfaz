// ws.ts — URL del WebSocket de inferencia. El servicio de transporte que la usa
// (captura de frames + 1-frame-en-vuelo + reconexion) llega en el slice 3
// (features/inference/services/videoStream.ts).

const wsBase = import.meta.env.VITE_WS_URL ?? 'ws://127.0.0.1:8000';

export const STREAM_URL = `${wsBase}/video_stream`;
