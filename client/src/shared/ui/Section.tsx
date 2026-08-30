// Section.tsx — seccion plegable de las columnas de Inferencia.
//
// Reemplaza el patron <SectionLabel>titulo</SectionLabel> + contenido: el rotulo pasa
// a ser el control que despliega y pliega lo que tiene a cargo. Nace de un problema
// concreto: cada cosecha de supervision le suma una seccion a la columna derecha
// (Render, Seguimiento, y las Zonas que vienen), y la columna no da mas.
//
// LA REGLA QUE HACE QUE ESTO NO SEA UNA TRAMPA: una seccion plegada tiene que seguir
// mostrando su ESTADO en el encabezado. Plegar es, literalmente, una maquina para
// fabricar el problema que este proyecto viene peleando —un control prendido sin
// efecto visible—: si "Seguimiento" se pliega con el suavizado encendido, el usuario
// ve las cajas arrastrarse y no tiene como saber por que. Por eso `estado`.
//
// Plegado INDEPENDIENTE, no acordeon: querer ver las metricas MIENTRAS se mueve el
// umbral es el caso normal, y un acordeon que cierra una para abrir otra se ve
// ordenado en el demo y estorba en el uso real.

import type { ReactNode } from 'react';
import { useUiStore, type SectionId } from '@/app/store/uiStore';
import { cn } from './cn';

interface SectionProps {
  id: SectionId;
  title: string;
  /**
   * Resumen de lo que la seccion tiene adentro, para leerlo SIN desplegarla
   * (el estilo activo, ON/OFF, los fps, la cantidad de errores).
   * Se muestra solo cuando esta plegada: abierta, los controles ya lo dicen.
   */
  estado?: ReactNode;
  /** Marca el estado como algo que reclama atencion (errores). */
  alerta?: boolean;
  /**
   * Seccion colgada DENTRO de otra (ej: Etiquetas dentro de Render). Se indenta con
   * la misma guia vertical que usa Interruptor para sus dependientes: si se dibujara
   * igual que su padre se leeria como hermana, no como hija.
   */
  anidada?: boolean;
  children: ReactNode;
  className?: string;
}

export function Section({ id, title, estado, alerta, anidada, children, className }: SectionProps) {
  const abierta = useUiStore((s) => s.sections[id]);
  const toggle = useUiStore((s) => s.toggleSection);

  const cuerpo = (
    <div className={cn('flex flex-col gap-2.5', abierta && className)}>
      <button
        type="button"
        aria-expanded={abierta}
        onClick={() => toggle(id)}
        className={cn(
          'group flex w-full items-center gap-2 rounded-[var(--radius-sm)] py-0.5 text-left',
          'transition-colors duration-150 focus-visible:outline-none',
        )}
      >
        <Chevron abierta={abierta} />
        <span className="lbl transition-colors group-hover:text-fg-muted">{title}</span>

        {/* El estado va a la derecha y solo cuando esta plegada: con la seccion
            abierta seria repetir lo que los controles ya muestran. */}
        {!abierta && estado != null && (
          <span
            className={cn(
              'ml-auto truncate font-mono text-[9px] font-semibold tracking-[0.1em]',
              alerta ? 'text-danger' : 'text-accent',
            )}
          >
            {estado}
          </span>
        )}
      </button>

      {/* Se DESMONTA al plegar, no se esconde por CSS: asi los paneles que hacen
          polling (metricas) dejan de pedirle datos al backend mientras nadie mira.
          El contador de errores es la excepcion y por eso vive en el encabezado,
          no adentro. */}
      {abierta && children}
    </div>
  );

  if (!anidada) return cuerpo;

  // Misma guia vertical + indentado que Interruptor.dependiente: el lenguaje de
  // "esto cuelga de aquello" ya existe en la piel, no hace falta inventar otro.
  return (
    <div className="flex items-stretch gap-2 pl-1.5">
      <span aria-hidden className="w-px shrink-0 bg-border" />
      <div className="min-w-0 flex-1">{cuerpo}</div>
    </div>
  );
}

function Chevron({ abierta }: { abierta: boolean }) {
  return (
    <svg
      viewBox="0 0 12 12"
      aria-hidden
      className={cn(
        'size-2.5 shrink-0 text-label transition-transform duration-150',
        abierta ? 'rotate-90' : 'rotate-0',
      )}
    >
      <path d="M4 2.5 L8 6 L4 9.5" fill="none" stroke="currentColor" strokeWidth="1.8"
            strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
