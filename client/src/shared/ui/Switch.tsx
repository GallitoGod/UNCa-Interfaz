// Switch.tsx — toggle accesible y controlado (role=switch). Es el primitivo base
// del ThemeToggle (el wiring de tema vive en el app shell, slice 2): aca solo el
// control presentacional, reutilizable para cualquier on/off.

import { cn } from './cn';

interface SwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string; // accesible (aria-label); no se dibuja salvo que el consumidor quiera
}

export function Switch({ checked, onChange, label }: SwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cn(
        'relative inline-flex h-5 w-9 shrink-0 items-center rounded-full',
        'transition-colors duration-200 focus-visible:outline-none',
        checked ? 'bg-accent' : 'bg-control border border-border',
      )}
    >
      <span
        className={cn(
          'inline-block size-4 rounded-full bg-white shadow-sm',
          'transition-transform duration-200 ease-[cubic-bezier(0.23,1,0.32,1)]',
          checked ? 'translate-x-4' : 'translate-x-0.5',
        )}
      />
    </button>
  );
}
