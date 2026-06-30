// uiStore.ts — estado de UI de primer nivel: vista activa.
// La navegacion es por estado (sin URLs; la app no las usa). El tema es unico
// (dark-only, piel Cabina Tecnica), asi que ya no hay estado de tema ni toggle.

import { create } from 'zustand';

export type View = 'inference' | 'models';

interface UiState {
  activeView: View;
  setView: (view: View) => void;
}

export const useUiStore = create<UiState>((set) => ({
  activeView: 'inference',
  setView: (view) => set({ activeView: view }),
}));
