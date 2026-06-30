// SectionLabel.tsx — encabezado de seccion en mono-mayuscula (la clase .lbl del
// spec). Es el rotulo de "lectura de instrumento" que separa grupos del panel.

import type { ReactNode } from 'react';
import { cn } from './cn';

export function SectionLabel({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={cn('lbl', className)}>{children}</div>;
}
