// ThemeToggle.tsx — wiring de tema: conecta el Switch presentacional (shared/ui)
// con el uiStore (que persiste y aplica la clase .dark). El primitivo no conoce
// el dominio; este componente si.

import { Switch } from '@/shared/ui/Switch';
import { useUiStore } from '../store/uiStore';

export function ThemeToggle() {
  const theme = useUiStore((s) => s.theme);
  const toggleTheme = useUiStore((s) => s.toggleTheme);

  return (
    <div className="flex items-center gap-2">
      <span className="select-none text-xs text-fg-muted">Oscuro</span>
      <Switch
        checked={theme === 'dark'}
        onChange={toggleTheme}
        label="Alternar modo oscuro"
      />
    </div>
  );
}
