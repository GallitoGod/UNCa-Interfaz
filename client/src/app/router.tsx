// router.tsx — ViewRouter: alterna entre las dos vistas de primer nivel segun el
// uiStore. Cada feature se carga PEREZOSAMENTE (code-splitting), reemplazando el
// flag modelsReady del scripts.js viejo.
//
// Inferencia se mantiene SIEMPRE montada (solo se oculta) para preservar la camara,
// el WebSocket y los refs del workspace al navegar a Modelos: la sesion no se
// desmonta, solo se pausa (ver useVisionSession + videoStream.pause/resume). Asi se
// cumple la regla del SDD 4.1.2 (pausar y reanudar sin reconectar ni repedir permisos).

import { lazy, Suspense } from 'react';
import { useUiStore } from './store/uiStore';
import { Spinner } from '@/shared/ui/Spinner';
import { cn } from '@/shared/ui/cn';

const InferenceView = lazy(() => import('@/features/inference/InferenceView'));
const ModelsView = lazy(() => import('@/features/models/ModelsView'));

function ViewFallback() {
  return (
    <div className="grid h-full place-items-center">
      <Spinner className="size-6" />
    </div>
  );
}

export function ViewRouter() {
  const view = useUiStore((s) => s.activeView);
  return (
    <Suspense fallback={<ViewFallback />}>
      {/* Inferencia nunca se desmonta: solo se oculta cuando la vista es Modelos. */}
      <div className={cn('h-full', view !== 'inference' && 'hidden')}>
        <InferenceView />
      </div>
      {view === 'models' && <ModelsView />}
    </Suspense>
  );
}
