// fields.tsx — campos controlados del wizard (reemplazan los helpers num/sel/chk/txt
// del configBuilder viejo). Puros: reciben value + onChange, no conocen el store.

const inputCls =
  'h-9 w-full rounded-[var(--radius-sm)] border border-border bg-control px-2 text-sm text-fg focus-visible:outline-none';

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
}: {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (v: T) => void;
}) {
  return (
    <label className="block space-y-1">
      <span className="text-xs font-medium text-fg-muted">{label}</span>
      <select
        className={inputCls}
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
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
      <h4 className="text-sm font-semibold text-fg">{title}</h4>
      {children}
    </section>
  );
}
