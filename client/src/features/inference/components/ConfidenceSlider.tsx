// ConfidenceSlider.tsx — umbral de confianza en vivo. Debounce para no inundar el
// backend (el viejo solo enviaba en 'change'; aca debounce en el arrastre).

import { useRef, useState, type CSSProperties } from 'react';
import { useUpdateConfidence } from '../hooks/useDiagnostics';
import { useStreamStore } from '../store/streamStore';

export function ConfidenceSlider() {
  const [value, setValue] = useState(50); // porcentaje [0,100]
  const update = useUpdateConfidence();
  const timer = useRef<number | undefined>(undefined);
  const resendStill = useStreamStore((s) => s.resendStill);

  function onChange(percent: number) {
    setValue(percent);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      // El backend lee el umbral en CADA inferencia, asi que con camara/video el
      // cambio se ve en el frame siguiente. Con una imagen fija hay que pedir
      // explicitamente una inferencia nueva, si no la pantalla queda con el
      // resultado viejo y parece que el umbral no se respeta.
      update.mutate(percent / 100, { onSuccess: () => resendStill() }); // backend espera [0,1]
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
