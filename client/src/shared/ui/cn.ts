// cn.ts — helper minimo para componer className condicionales sin dependencias.
// Filtra falsy y une con espacio. Para casos simples alcanza; si despues hace
// falta merge inteligente de clases Tailwind se puede cambiar por clsx+twMerge.

export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ');
}
