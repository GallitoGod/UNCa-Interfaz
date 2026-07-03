// Header.tsx — title bar: marca, navegacion de vistas (segmentado) y el modelo
// activo (solo lectura). La seleccion de modelo vive en el panel izquierdo de
// Inferencia; aca solo se refleja cual esta cargado.
//
// Firma visual: marca de reticula/apertura junto al wordmark (dominio camara/optica)
// con "Lens" en cian. El acento cian aparece solo en la pestana activa y en el dato
// del modelo (color = recurso escaso).

import { useUiStore, type View } from '../store/uiStore';
import { useWorkspaceStore } from '@/features/vision-workspace/store/workspaceStore';
import { Tabs } from '@/shared/ui/Tabs';

const NAV: { key: View; label: string }[] = [
  { key: 'inference', label: 'Inferencia' },
  { key: 'models', label: 'Modelos' },
];

// Etiqueta corta del tipo de modelo para el badge del title bar.
const TYPE_LABEL: Record<string, string> = {
  detection: 'DET',
  classification: 'CLS',
  segmentation: 'SEG',
};

// Marca de apertura/reticula: el signo del dominio (visor de camara).
function ReticleMark() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-accent">
      <circle cx="12" cy="12" r="8.5" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle cx="12" cy="12" r="2" fill="currentColor" />
    </svg>
  );
}

export function Header() {
  const activeView = useUiStore((s) => s.activeView);
  const setView = useUiStore((s) => s.setView);
  const activeModel = useWorkspaceStore((s) => s.activeModel);

  return (
    <header className="flex h-14 shrink-0 items-center gap-6 border-b border-border bg-surface px-5">
      {/* Marca */}
      <div className="flex items-center gap-2">
        <ReticleMark />
        <span className="text-sm font-semibold tracking-tight">
          UNCa<span className="text-accent">Lens</span>
        </span>
      </div>

      {/* Navegacion de vistas (control segmentado) */}
      <Tabs aria-label="Vista" tabs={NAV} value={activeView} onChange={setView} />

      {/* Modelo activo (solo lectura): nombre mono + badge de tipo. */}
      <div className="ml-auto flex items-center gap-2">
        {activeModel ? (
          <>
            <span className="lbl">Modelo</span>
            <span className="font-mono text-xs text-fg">{activeModel.name}</span>
            <span className="rounded-[var(--radius-sm)] bg-accent-soft px-1.5 py-0.5 font-mono text-[9px] font-semibold text-accent">
              {TYPE_LABEL[activeModel.type] ?? activeModel.type}
            </span>
          </>
        ) : (
          <span className="font-mono text-xs text-fg-subtle">Sin modelo</span>
        )}
      </div>
    </header>
  );
}
