// Tabs.tsx — tablist accesible y controlado. Reporta la pestana activa; el panel
// lo renderiza el consumidor segun `value` (separa seleccion de contenido).
//
// Contrato de comportamiento (hand-roll con el contrato completo, ya que no hay
// libreria headless en el proyecto): roles tablist/tab, seleccion controlada,
// roving tabindex y navegacion por teclado (flechas + Home/End).

import { useRef, type KeyboardEvent } from 'react';
import { cn } from './cn';

export interface TabItem<K extends string = string> {
  key: K;
  label: string;
}

interface TabsProps<K extends string> {
  tabs: TabItem<K>[];
  value: K;
  onChange: (key: K) => void;
  'aria-label'?: string;
}

export function Tabs<K extends string>({
  tabs,
  value,
  onChange,
  'aria-label': ariaLabel,
}: TabsProps<K>) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  function onKeyDown(e: KeyboardEvent<HTMLButtonElement>, index: number) {
    let next = index;
    if (e.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (e.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = tabs.length - 1;
    else return;
    e.preventDefault();
    onChange(tabs[next].key);
    refs.current[next]?.focus();
  }

  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className="inline-flex gap-1 rounded-[var(--radius-md)] bg-control p-1"
    >
      {tabs.map((tab, i) => {
        const active = tab.key === value;
        return (
          <button
            key={tab.key}
            ref={(el) => {
              refs.current[i] = el;
            }}
            role="tab"
            type="button"
            aria-selected={active}
            // Roving tabindex: solo la pestana activa es tabbable.
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(tab.key)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={cn(
              'rounded-[var(--radius-sm)] px-3 py-1.5 text-sm font-medium',
              'transition-colors duration-150 focus-visible:outline-none',
              active
                ? 'bg-surface-raised text-fg shadow-sm'
                : 'text-fg-muted hover:text-fg',
            )}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
