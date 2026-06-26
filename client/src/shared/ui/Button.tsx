// Button.tsx — boton base de la app. Centraliza variantes y estados (hover,
// active, focus, disabled) para no repetir strings de Tailwind por todos lados.
// Extiende <button> nativo: hereda type, disabled, onClick, aria, etc.

import type { ButtonHTMLAttributes } from 'react';
import { cn } from './cn';

type Variant = 'primary' | 'outline' | 'ghost' | 'danger';
type Size = 'sm' | 'md';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-accent text-accent-fg hover:bg-accent-hover',
  outline: 'border border-border text-fg hover:bg-control hover:border-border-strong',
  ghost: 'text-fg-muted hover:bg-control hover:text-fg',
  danger: 'bg-danger text-white hover:opacity-90',
};

const SIZES: Record<Size, string> = {
  sm: 'h-8 px-3 text-sm gap-1.5',
  md: 'h-9 px-4 text-sm gap-2',
};

export function Button({
  variant = 'outline',
  size = 'md',
  className,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center rounded-[var(--radius-sm)] font-medium',
        'transition-[background-color,border-color,color,transform] duration-150',
        'focus-visible:outline-none active:scale-[0.98]',
        'disabled:pointer-events-none disabled:opacity-50',
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...props}
    />
  );
}
