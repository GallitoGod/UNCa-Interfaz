// ConfidenceSlider.tsx — umbral de confianza en vivo. Debounce para no inundar el
// backend (el viejo solo enviaba en 'change'; aca debounce en el arrastre).

import { useRef, useState, type CSSProperties } from 'react';
import { useUpdateConfidence } from '../hooks/useDiagnostics';

export function ConfidenceSlider() {
  const [value, setValue] = useState(50); // porcentaje [0,100]
  const update = useUpdateConfidence();
  const timer = useRef<number | undefined>(undefined);

  function onChange(percent: number) {
    setValue(percent);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      update.mutate(percent / 100); // backend espera [0,1]
    }, 200);
  }

  return (
    <div className="space-y-2.5">
      <div className="flex items-baseline justify-between">
        <span className="text-[12.5px] font-medium text-fg-muted">Confianza</span>
        <span className="font-mono text-xs font-semibold text-accent">{value}%</span>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        // --pct controla el fill cian del track (regla .range-cyan en index.css).
        style={{ '--pct': `${value}%` } as CSSProperties}
        className="range-cyan"
        aria-label="Umbral de confianza"
      />
    </div>
  );
}
