// Badge.tsx — pill de estado con punto. Variantes del spec:
//   live   -> EN VIVO: rojo tenue, punto que pulsa (cadencia lenta)
//   rec    -> REC: borde rojo sobre fondo oscuro, punto que pulsa (cadencia rapida)
//   active -> ACTIVO: verde con glow, sin pulso
// El punto pulsa con la animacion recpulse (definida en index.css).

import type { ReactNode } from 'react';
import { cn } from './cn';

type Variant = 'live' | 'rec' | 'active';

const WRAP: Record<Variant, string> = {
  live: 'bg-[rgba(255,77,79,0.13)] text-[#ff7a7c]',
  rec: 'bg-[rgba(5,8,12,0.72)] border border-[rgba(255,77,79,0.35)] text-[#ff7a7c]',
  active: 'bg-accent-soft text-[#5fd07a]',
};

// Color + glow + cadencia del punto por variante.
const DOT: Record<Variant, string> = {
  live: 'bg-danger [animation:recpulse_1.4s_infinite]',
  rec: 'bg-danger [animation:recpulse_1s_infinite]',
  active: 'bg-success shadow-[0_0_7px_#28c840]',
};

export function Badge({
  variant,
  children,
}: {
  variant: Variant;
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full px-2.5 py-1',
        'font-mono text-[10px] font-bold uppercase tracking-wide',
        WRAP[variant],
      )}
    >
      <span className={cn('size-1.5 rounded-full', DOT[variant])} />
      {children}
    </span>
  );
}
