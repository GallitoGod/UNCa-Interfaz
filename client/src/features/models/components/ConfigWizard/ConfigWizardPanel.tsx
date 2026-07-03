// ConfigWizardPanel.tsx — carga la config existente (GET /configs/{name}) o el template
// (backend), inicializa el wizardStore y renderiza el ConfigWizard. Maneja la carga.

import { useEffect, useRef } from 'react';
import { Spinner } from '@/shared/ui/Spinner';
import { useModelConfig, useConfigTemplate } from '../../hooks/useModelsList';
import { useWizardStore } from '../../store/wizardStore';
import { ConfigWizard } from './ConfigWizard';

export function ConfigWizardPanel({ file, onClose }: { file: string; onClose: () => void }) {
  const baseName = file.replace(/\.[^.]+$/, '');
  const cfgQuery = useModelConfig(baseName); // GET /configs/{name}: config existente (o null)
  const tplQuery = useConfigTemplate('detection'); // template por defecto (modelo nuevo)

  const init = useWizardStore((s) => s.init);
  const reset = useWizardStore((s) => s.reset);
  const initialized = useRef(false);

  const ready = !cfgQuery.isLoading && !tplQuery.isLoading;

  useEffect(() => {
    if (!ready || initialized.current) return;
    // cfgQuery.data: ModelConfig si existe, null si el modelo no tiene config, y
    // undefined si la lectura fallo (config corrupta -> 500). En todos los casos sin
    // config usable caemos al template del backend (regla SDD 4.1.4: no bloquear).
    const existing = cfgQuery.data ?? null;
    const initial = existing ?? tplQuery.data?.config ?? null;
    if (initial) {
      init(file, initial);
      initialized.current = true;
    }
  }, [ready, cfgQuery.data, tplQuery.data, file, init]);

  // Al cerrar/cambiar de modelo, limpiar el estado del wizard.
  useEffect(() => () => reset(), [reset]);

  if (!ready) {
    return (
      <div className="grid place-items-center py-12">
        <Spinner className="size-6" />
      </div>
    );
  }

  // El template es el fallback universal: si ni siquiera ese cargo, no hay con que armar
  // el wizard (backend caido). La config propia es opcional, no bloquea.
  if (!initialized.current && !tplQuery.data) {
    return (
      <p className="py-8 text-center text-sm text-danger">
        No se pudo cargar el template de config (backend caido). Verifica que uvicorn este corriendo.
      </p>
    );
  }

  return <ConfigWizard onClose={onClose} initialAnchorDefaults={tplQuery.data?.anchor_defaults ?? null} />;
}
