// RenderSettings.tsx — panel de dibujo del feed (columna derecha de Inferencia).
//
// Va ACA y no en el modal de "Configuracion avanzada" a proposito: estos controles
// se usan mirando el feed —prender, ver el efecto, apagar— y el flujo
// guardar-y-cerrar del modal pelea con eso. En el modal se quedan los ajustes de
// configurar-y-olvidar (los colores). La regla: el control tiene que estar donde el
// usuario puede ver el efecto AL MISMO TIEMPO que lo toca.
//
// Los tres ajustes son del USUARIO (no del modelo): persisten en localStorage via
// workspaceStore y se empujan al backend, que es el que dibuja desde el 2026-08-26.
// Se aplican al frame SIGUIENTE.

import { useWorkspaceStore } from '@/features/vision-workspace/store/workspaceStore';
import type { BoxStyle } from '@/features/vision-workspace/services/types';
import { pushDrawSettings } from '../api/drawSettings';
import { useStreamStore } from '../store/streamStore';
import { cn } from '@/shared/ui/cn';

// Cuatro estilos de una familia de doce: cada uno resuelve un caso real y el resto
// seria ruido (el wizard de modelos ya tuvo que podar por lo mismo en junio).
const ESTILOS: { key: BoxStyle; label: string; hint: string }[] = [
  { key: 'box', label: 'Caja', hint: 'Rectangulo completo' },
  { key: 'round', label: 'Redonda', hint: 'Rectangulo con esquinas redondeadas' },
  { key: 'corner', label: 'Esquinas', hint: 'Solo las esquinas: deja ver la imagen debajo' },
  { key: 'dot', label: 'Punto', hint: 'Un punto por deteccion: para muchas cajas chicas' },
];

export function RenderSettings() {
  const drawSettings = useWorkspaceStore((s) => s.drawSettings);
  const setDrawSettings = useWorkspaceStore((s) => s.setDrawSettings);
  const resendStill = useStreamStore((s) => s.resendStill);

  // Aplica el cambio, lo empuja al backend y re-infiere si la fuente es una imagen
  // fija (sin esto el usuario cambia el estilo y no pasa nada en pantalla: no hay
  // frame siguiente que dibujar). Camara y video refrescan solos.
  function aplicar(patch: Partial<typeof drawSettings>) {
    const next = { ...drawSettings, ...patch };
    setDrawSettings(patch);
    pushDrawSettings(next);
    resendStill();
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-1.5" role="group" aria-label="Estilo de marca">
        {ESTILOS.map((e) => {
          const activo = drawSettings.boxStyle === e.key;
          return (
            <button
              key={e.key}
              type="button"
              title={e.hint}
              aria-pressed={activo}
              onClick={() => aplicar({ boxStyle: e.key })}
              className={cn(
                'rounded-[var(--radius-sm)] border px-2 py-1.5 text-xs font-medium',
                'transition-colors duration-150 focus-visible:outline-none active:scale-[0.98]',
                activo
                  ? 'border-accent-border bg-accent-soft text-accent'
                  : 'border-border bg-control text-fg-subtle hover:text-fg hover:border-border-strong',
              )}
            >
              {e.label}
            </button>
          );
        })}
      </div>

      <Interruptor
        label="Etiquetas que se esquivan"
        hint="Corre los carteles para que no se tapen entre si. Cuesta un poco mas con muchas cajas."
        on={drawSettings.smartLabels}
        onToggle={() => aplicar({ smartLabels: !drawSettings.smartLabels })}
      />
      <Interruptor
        label="Grosor automatico"
        hint="Deriva el grosor y el tamano del texto de la resolucion del frame."
        on={drawSettings.autoScale}
        onToggle={() => aplicar({ autoScale: !drawSettings.autoScale })}
      />
    </div>
  );
}

// Interruptor de una linea: etiqueta a la izquierda, estado mono a la derecha. Sin
// componente Switch porque el re-skin "Cabina Tecnica" lo elimino a proposito; el
// estado se lee del color y del texto, como el resto de la piel.
function Interruptor({
  label,
  hint,
  on,
  onToggle,
}: {
  label: string;
  hint: string;
  on: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      title={hint}
      onClick={onToggle}
      className={cn(
        'flex w-full items-center justify-between gap-2 rounded-[var(--radius-sm)] border',
        'px-2.5 py-2 text-left text-xs transition-colors duration-150',
        'focus-visible:outline-none active:scale-[0.99]',
        on
          ? 'border-accent-border bg-accent-soft text-fg'
          : 'border-border bg-control text-fg-subtle hover:text-fg hover:border-border-strong',
      )}
    >
      <span className="leading-tight">{label}</span>
      <span
        className={cn(
          'shrink-0 font-mono text-[9px] font-semibold tracking-[0.1em]',
          on ? 'text-accent' : 'text-label',
        )}
      >
        {on ? 'ON' : 'OFF'}
      </span>
    </button>
  );
}
