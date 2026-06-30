// fields.tsx — campos controlados del wizard (reemplazan los helpers num/sel/chk/txt
// del configBuilder viejo). Puros: reciben value + onChange, no conocen el store.

const inputCls =
  'h-9 w-full rounded-[var(--radius-sm)] border border-border bg-control px-2 text-sm text-fg focus-visible:outline-none focus-visible:border-accent';

export function NumberField({
  label,
  value,
  onChange,
  step,
  min,
}: {
  label: string;
  value: number | null;
  onChange: (v: number | null) => void;
  step?: number;
  min?: number;
}) {
  return (
    <label className="block space-y-1">
      <span className="text-xs font-medium text-fg-muted">{label}</span>
      <input
        type="number"
        className={inputCls}
        value={value ?? ''}
        step={step}
        min={min}
        onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
      />
    </label>
  );
}

export function TextField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block space-y-1">
      <span className="text-xs font-medium text-fg-muted">{label}</span>
      <input
        type="text"
        className={inputCls}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

export function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
  labels,
  disabled,
}: {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (v: T) => void;
  // Etiquetas legibles opcionales por valor (si falta, se muestra el valor crudo).
  labels?: Partial<Record<T, string>>;
  disabled?: boolean;
}) {
  return (
    <label className="block space-y-1">
      <span className="text-xs font-medium text-fg-muted">{label}</span>
      <select
        className={inputCls}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {labels?.[o] ?? o}
          </option>
        ))}
      </select>
    </label>
  );
}

export function CheckField({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2">
      <input
        type="checkbox"
        className="size-4 accent-[var(--color-accent)]"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="text-sm text-fg">{label}</span>
    </label>
  );
}

export function FieldGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <h4 className="lbl">{title}</h4>
      {children}
    </section>
  );
}

// ColorField — color picker que mapea un [r,g,b] (0-255) al <input type=color> (hex).
// Si el valor es null cae al gris de padding por defecto (114,114,114).
const DEFAULT_PAD: [number, number, number] = [114, 114, 114];

function rgbToHex([r, g, b]: number[]): string {
  const h = (n: number) => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, '0');
  return `#${h(r)}${h(g)}${h(b)}`;
}

function hexToRgb(hex: string): number[] {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) return [...DEFAULT_PAD];
  return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)];
}

export function ColorField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number[] | null;
  onChange: (v: number[]) => void;
}) {
  const rgb = value && value.length >= 3 ? value : [...DEFAULT_PAD];
  return (
    <label className="block space-y-1">
      <span className="text-xs font-medium text-fg-muted">{label}</span>
      <div className="flex items-center gap-2">
        <input
          type="color"
          className="h-9 w-12 cursor-pointer rounded-[var(--radius-sm)] border border-border bg-control p-1"
          value={rgbToHex(rgb)}
          onChange={(e) => onChange(hexToRgb(e.target.value))}
        />
        <span className="font-mono text-xs text-fg-muted">
          {rgb.map((n) => Math.round(n)).join(', ')}
        </span>
      </div>
    </label>
  );
}

// AdvancedSection — bloque colapsable (cerrado por defecto) para parametros que el caso
// comun no toca. Reusa <details>/<summary> nativos (accesibles sin JS extra).
export function AdvancedSection({
  title = 'Avanzado',
  defaultOpen = false,
  children,
}: {
  title?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  return (
    <details open={defaultOpen} className="group rounded-[var(--radius-md)] border border-border bg-control/40">
      <summary className="lbl cursor-pointer list-none px-3 py-2 marker:content-none">
        <span className="select-none">▸ {title}</span>
      </summary>
      <div className="space-y-3 px-3 pb-3 pt-1">{children}</div>
    </details>
  );
}
