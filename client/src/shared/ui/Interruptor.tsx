// Interruptor.tsx — toggle de una linea: etiqueta a la izquierda, estado mono a la
// derecha.
//
// Nacio privado dentro de RenderSettings (2026-08-27) y se mudo aca cuando el panel
// de Seguimiento necesito el mismo control. No es el `Switch` que el re-skin "Cabina
// Tecnica" elimino a proposito: no hay riel ni perilla, el estado se lee del color y
// del texto, como el resto de la piel.

import { cn } from './cn';

interface InterruptorProps {
  label: string;
  hint: string;
  on: boolean;
  onToggle: () => void;
  /** Apagado y no operable. El motivo va en `hint`, que es lo que el usuario lee. */
  disabled?: boolean;
  /**
   * Marca el control como SUBORDINADO a otro (ej: el suavizado depende del
   * seguimiento). Se indenta y se marca con una guia vertical: la dependencia tiene
   * que verse en la disposicion, no adivinarse leyendo el tooltip.
   */
  dependiente?: boolean;
}

export function Interruptor({
  label,
  hint,
  on,
  onToggle,
  disabled = false,
  dependiente = false,
}: InterruptorProps) {
  const boton = (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      disabled={disabled}
      title={hint}
      onClick={onToggle}
      className={cn(
        'flex w-full items-center justify-between gap-2 rounded-[var(--radius-sm)] border',
        'px-2.5 py-2 text-left text-xs transition-colors duration-150',
        'focus-visible:outline-none active:scale-[0.99]',
        // El :disabled queda fuera de la regla de cursor:pointer de index.css a
        // proposito, asi que el cursor ya avisa solo que no se puede tocar.
        disabled
          ? 'cursor-not-allowed border-border bg-control text-label opacity-50 active:scale-100'
          : on
            ? 'border-accent-border bg-accent-soft text-fg'
            : 'border-border bg-control text-fg-subtle hover:text-fg hover:border-border-strong',
      )}
    >
      <span className="leading-tight">{label}</span>
      <span
        className={cn(
          'shrink-0 font-mono text-[9px] font-semibold tracking-[0.1em]',
          on && !disabled ? 'text-accent' : 'text-label',
        )}
      >
        {on ? 'ON' : 'OFF'}
      </span>
    </button>
  );

  if (!dependiente) return boton;

  // Guia vertical + indentado: el control colgado se lee como colgado de un vistazo.
  return (
    <div className="flex items-stretch gap-2 pl-1.5">
      <span aria-hidden className="w-px shrink-0 bg-border" />
      <div className="min-w-0 flex-1">{boton}</div>
    </div>
  );
}
