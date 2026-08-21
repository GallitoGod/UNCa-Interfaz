// workspaceStore.ts — estado del vision-workspace: modelo activo (name + type) y
// los ajustes de dibujo. El type del modelo activo es lo que usa el render para
// enrutar a la estrategia correcta.

import { create } from 'zustand';
import type { ModelType } from '@/shared/api/types';
import type { DrawSettings } from '../services/types';

interface WorkspaceState {
  activeModel: { name: string; type: ModelType } | null;
  /**
   * Modelo que se esta armando en el backend ahora mismo (null = ninguno). Cargar
   * un modelo no es instantaneo -sesion del runtime + warmup- y durante ese rato el
   * canvas mostraba el frame viejo como si nada, dando la sensacion de que la app se
   * colgo. El workspace lo lee para tapar el feed con el cartel de carga.
   */
  loadingModel: string | null;
  drawSettings: DrawSettings;
  setActiveModel: (name: string, type: ModelType) => void;
  clearActiveModel: () => void;
  setLoadingModel: (name: string | null) => void;
  setDrawSettings: (patch: Partial<DrawSettings>) => void;
}

// Default de bbox cian (coherente con el #00BFFF historico); label oscuro legible
// sobre el fondo cian de la etiqueta.
const DEFAULT_DRAW_SETTINGS: DrawSettings = {
  bboxColor: '#00BFFF',
  labelColor: '#001018',
  maskAlpha: 0.5,
};

// Persistencia en localStorage (mismo patron manual que uiStore, sin middleware).
// SDD 4.1.3: los colores de dibujo deben sobrevivir entre sesiones.
const DRAW_KEY = 'uncalens-draw-settings';

function readStoredDrawSettings(): DrawSettings {
  try {
    const raw = localStorage.getItem(DRAW_KEY);
    if (!raw) return DEFAULT_DRAW_SETTINGS;
    const parsed = JSON.parse(raw) as Partial<DrawSettings>;
    // Merge sobre los defaults: tolera versiones viejas sin claves nuevas
    // (ej: maskAlpha/colormap agregados despues).
    return { ...DEFAULT_DRAW_SETTINGS, ...parsed };
  } catch {
    return DEFAULT_DRAW_SETTINGS;
  }
}

export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  activeModel: null,
  loadingModel: null,
  drawSettings: readStoredDrawSettings(),

  setActiveModel: (name, type) => set({ activeModel: { name, type } }),
  clearActiveModel: () => set({ activeModel: null }),
  setLoadingModel: (loadingModel) => set({ loadingModel }),
  setDrawSettings: (patch) =>
    set((s) => {
      const next = { ...s.drawSettings, ...patch };
      try {
        localStorage.setItem(DRAW_KEY, JSON.stringify(next));
      } catch {
        // localStorage lleno/deshabilitado: el cambio sigue valiendo en memoria.
      }
      return { drawSettings: next };
    }),
}));
