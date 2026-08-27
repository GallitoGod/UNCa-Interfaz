// RenderSettings.tsx — panel de dibujo del feed (columna derecha de Inferencia).
//
// Va ACA y no en el modal de "Configuracion avanzada" a proposito: estos controles
// se usan mirando el feed —prender, ver el efecto, apagar— y el flujo
// guardar-y-cerrar del modal pelea con eso. En el modal se quedan los ajustes de
// configurar-y-olvidar (los colores). La regla: el control tiene que estar donde el
// usuario puede ver el efecto AL MISMO TIEMPO que lo toca.
//
// Los ajustes son del USUARIO (no del modelo): persisten en localStorage via
// workspaceStore y se empujan al backend, que es el que dibuja desde el 2026-08-26.
// Se aplican al frame SIGUIENTE.
//
// Lo que NO va aca: seguimiento, suavizado y trazas (TrackingSettings.tsx). Esos
// gobiernan como se sigue un objeto A LO LARGO DEL TIEMPO, tienen dependencias entre
// si y solo valen para camara y video; agrupados aparte, esas reglas se aplican al
// bloque entero en vez de repetirse control por control.

import { useWorkspaceStore } from '@/features/vision-workspace/store/workspaceStore';
import type { BoxStyle, ModelType } from '@/features/vision-workspace/services/types';
import { Interruptor } from '@/shared/ui/Interruptor';
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

// Tipos cuyo resultado se DIBUJA sobre el frame (output_kind="frame" en el backend).
// Un clasificador devuelve TEXTO: el backend no compone nada, manda el envelope JSON y
// el cliente pinta su propio frame con un panel HTML encima. Ninguno de estos controles
// puede cambiarle un pixel, asi que mostrarlos seria ofrecer perillas desconectadas.
const TIPOS_QUE_SE_DIBUJAN: ModelType[] = ['detection', 'segmentation'];

/**
 * Si el panel de render tiene algo que gobernar para el modelo activo.
 *
 * Sin modelo (null) devuelve true a proposito: son ajustes del USUARIO, persistidos, y
 * dejarlos a mano antes de cargar nada es legitimo. La columna solo se poda cuando hay
 * un modelo cargado que no dibuja.
 */
export function panelDeRenderAplica(type: ModelType | null | undefined): boolean {
  return type == null || TIPOS_QUE_SE_DIBUJAN.includes(type);
}

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
        label="Sombreado"
        hint="Rellena la caja con el color de acento translucido. Se lleva bien con Esquinas; con muchas cajas superpuestas los rellenos se suman y tapan la imagen."
        on={drawSettings.shading}
        onToggle={() => aplicar({ shading: !drawSettings.shading })}
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

