// ModelLoadingOverlay.tsx — tapa el feed mientras el backend arma el pipeline de un
// modelo. Cargar no es instantaneo (sesion del runtime + warmup, varios segundos en
// el clasificador) y antes el canvas seguia mostrando el frame viejo: parecia que la
// app se habia colgado. Acto seguido: ocupar la pantalla y decir que se esta haciendo.
//
// Pulso con la animacion 'recpulse' de index.css, la misma de los badges EN VIVO/REC:
// el "estoy trabajando" de la app ya tiene un idioma visual, no se inventa otro.

interface ModelLoadingOverlayProps {
  /** Nombre del modelo que se esta armando. */
  name: string;
}

export function ModelLoadingOverlay({ name }: ModelLoadingOverlayProps) {
  return (
    <div
      className="absolute inset-0 z-10 grid place-items-center bg-feed/95 backdrop-blur-[2px]"
      role="status"
      aria-live="polite"
    >
      <div className="flex flex-col items-center gap-3">
        <span
          className="font-mono text-[13px] font-semibold uppercase tracking-[0.34em] text-accent"
          style={{ animation: 'recpulse 1.2s ease-in-out infinite' }}
        >
          Armando hot path
        </span>

        {/* Barra de progreso indeterminada: no sabemos cuanto falta (depende del
            runtime y del warmup), asi que se muestra actividad, no porcentaje. */}
        <div className="h-px w-56 overflow-hidden bg-border">
          <div
            className="h-full w-1/3 bg-accent"
            style={{ animation: 'hotpath-scan 1.4s ease-in-out infinite' }}
          />
        </div>

        <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-label">
          {name}
        </span>
      </div>
    </div>
  );
}
