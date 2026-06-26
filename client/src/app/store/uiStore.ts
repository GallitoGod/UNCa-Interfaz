// uiStore.ts — estado de UI de primer nivel: vista activa y tema.
// La navegacion es por estado (sin URLs; la app no las usa). El tema persiste en
// localStorage y se refleja como la clase .dark en <html> (la que leen los tokens).

import { create } from 'zustand';

export type Theme = 'dark' | 'light';
export type View = 'inference' | 'models';

const THEME_KEY = 'uncalens-theme';

function readStoredTheme(): Theme {
  const saved = localStorage.getItem(THEME_KEY);
  return saved === 'light' || saved === 'dark' ? saved : 'dark'; // default: oscuro
}

function applyTheme(theme: Theme): void {
  document.documentElement.classList.toggle('dark', theme === 'dark');
}

// Aplica el tema guardado al cargar el modulo (antes del primer render) para que
// no haya parpadeo si el usuario habia elegido claro.
const initialTheme = readStoredTheme();
applyTheme(initialTheme);

interface UiState {
  activeView: View;
  theme: Theme;
  setView: (view: View) => void;
  toggleTheme: () => void;
}

export const useUiStore = create<UiState>((set, get) => ({
  activeView: 'inference',
  theme: initialTheme,

  setView: (view) => set({ activeView: view }),

  toggleTheme: () => {
    const next: Theme = get().theme === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    localStorage.setItem(THEME_KEY, next);
    set({ theme: next });
  },
}));
