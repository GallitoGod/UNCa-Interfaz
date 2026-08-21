// streamStore.ts — fuente de video activa + estado del stream. Los componentes de
// fuente (CameraSource/FileSource) escriben la fuente; el orquestador
// (useVisionSession) reacciona y maneja el media + el WS.

import { create } from 'zustand';
import type { StreamStatus } from '../services/videoStream';

export type Source =
  | { kind: 'none' }
  | { kind: 'camera'; deviceId: string }
  | { kind: 'file-video'; url: string }
  | { kind: 'file-image'; url: string };

interface StreamState {
  source: Source;
  status: StreamStatus;
  lastError: string | null;
  /**
   * Contador de "volve a inferir el frame actual". Solo aplica a fuentes ESTATICAS
   * (imagen): con camara o video ya llega un frame nuevo cada tick y el cambio se ve
   * solo. Con una imagen el envio es one-shot, asi que cambiar de modelo o mover el
   * umbral no producia ninguna inferencia nueva y la pantalla quedaba mostrando el
   * resultado viejo. Quien cambia un parametro llama a resendStill().
   */
  stillNonce: number;
  setCameraSource: (deviceId: string) => void;
  setFileVideo: (url: string) => void;
  setFileImage: (url: string) => void;
  clearSource: () => void;
  setStatus: (status: StreamStatus) => void;
  setError: (error: string | null) => void;
  resendStill: () => void;
}

export const useStreamStore = create<StreamState>((set) => ({
  source: { kind: 'none' },
  status: 'closed',
  lastError: null,
  stillNonce: 0,

  setCameraSource: (deviceId) =>
    set({ source: { kind: 'camera', deviceId }, lastError: null }),
  setFileVideo: (url) => set({ source: { kind: 'file-video', url }, lastError: null }),
  setFileImage: (url) => set({ source: { kind: 'file-image', url }, lastError: null }),
  clearSource: () => set({ source: { kind: 'none' } }),

  setStatus: (status) => set({ status }),
  setError: (lastError) => set({ lastError }),

  resendStill: () => set((s) => ({ stillNonce: s.stillNonce + 1 })),
}));
