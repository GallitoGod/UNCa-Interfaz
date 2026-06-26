// Header.tsx — chrome superior: marca, navegacion de vistas, slot del selector de
// modelo y el toggle de tema. La navegacion escribe en el uiStore.
//
// Firma visual: una marca de reticula/apertura junto al wordmark (dominio camara/
// optica). El acento cian aparece solo en la pestana activa (color = scarce resource).

import { type ReactNode } from 'react';
import { useUiStore, type View } from '../store/uiStore';
import { ThemeToggle } from './ThemeToggle';
import { cn } from '@/shared/ui/cn';

const NAV: { view: View; label: string }[] = [
  { view: 'inference', label: 'Inferencia' },
  { view: 'models', label: 'Modelos' },
];

// Marca de apertura/reticula: el signo del dominio (visor de camara).
function ReticleMark() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-accent">
      <circle cx="12" cy="12" r="8.5" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle cx="12" cy="12" r="2" fill="currentColor" />
    </svg>
  );
}

export function Header({ modelSelector }: { modelSelector?: ReactNode }) {
  const activeView = useUiStore((s) => s.activeView);
  const setView = useUiStore((s) => s.setView);

  return (
    <header className="flex h-14 shrink-0 items-center gap-6 border-b border-border bg-canvas px-5">
      {/* Marca */}
      <div className="flex items-center gap-2">
        <ReticleMark />
        <span className="text-sm font-semibold tracking-tight">UNCaLens</span>
      </div>

      {/* Navegacion de vistas */}
      <nav className="flex items-center gap-1">
        {NAV.map(({ view, label }) => {
          const active = view === activeView;
          return (
            <button
              key={view}
              type="button"
              aria-current={active ? 'page' : undefined}
              onClick={() => setView(view)}
              className={cn(
                'rounded-[var(--radius-sm)] px-3 py-1.5 text-sm font-medium',
                'transition-colors duration-150 focus-visible:outline-none',
                active ? 'bg-control text-fg' : 'text-fg-muted hover:text-fg',
              )}
            >
              {label}
            </button>
          );
        })}
      </nav>

      {/* Empuja el cluster derecho */}
      <div className="ml-auto flex items-center gap-4">
        {/* Slot del selector de modelo (lo provee la feature inference, slice 3) */}
        {modelSelector}
        <ThemeToggle />
      </div>
    </header>
  );
}
