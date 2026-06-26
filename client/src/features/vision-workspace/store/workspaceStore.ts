// workspaceStore.ts — estado del vision-workspace: modelo activo (name + type) y
// los ajustes de dibujo. El type del modelo activo es lo que usa el render para
// enrutar a la estrategia correcta.

import { create } from 'zustand';
import type { ModelType } from '@/shared/api/types';
import type { DrawSettings } from '../services/types';

interface WorkspaceState {
  activeModel: { name: string; type: ModelType } | null;
  drawSettings: DrawSettings;
  setActiveModel: (name: string, type: ModelType) => void;
  clearActiveModel: () => void;
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
  drawSettings: readStoredDrawSettings(),

  setActiveModel: (name, type) => set({ activeModel: { name, type } }),
  clearActiveModel: () => set({ activeModel: null }),
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
