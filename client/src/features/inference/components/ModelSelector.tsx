// ModelSelector.tsx — lista de modelos cargables del panel izquierdo de inferencia.
// Selecciona el modelo a cargar en el backend y fija el modelo activo del workspace
// (cuyo type enruta la estrategia de presentacion). Misma logica que antes; ahora se
// presenta como lista de ModelRow en vez de un <select>.

import { useEffect, useRef } from 'react';
import type { ModelType } from '@/shared/api/types';
import { useModels, useSelectModel } from '../hooks/useModels';
import { getModelType } from '../api/models';
import { useWorkspaceStore } from '@/features/vision-workspace/store/workspaceStore';
import { ModelRow } from './ModelRow';

export function ModelSelector() {
  const { data: models, isLoading } = useModels();
  const selectModel = useSelectModel();
  const activeModel = useWorkspaceStore((s) => s.activeModel);
  const setActiveModel = useWorkspaceStore((s) => s.setActiveModel);
  const autoSelected = useRef(false);

  async function handleSelect(name: string) {
    if (!name) return;
    try {
      await selectModel.mutateAsync(name);
      // Leer el model_type real del config (GET /configs/{name}) para enrutar la
      // estrategia del workspace. Si no se puede leer, se asume 'detection' (es el
      // unico tipo cargable hoy; CLS/SEG aun dan 501 al cargar en el backend).
      let type: ModelType = 'detection';
      try {
        const real = await getModelType(name);
        if (real) type = real;
      } catch (e) {
        console.warn('No se pudo leer el model_type del config, se asume detection:', e);
      }
      setActiveModel(name, type);
    } catch (err) {
      console.error('No se pudo seleccionar el modelo:', err);
    }
  }

  // Auto-seleccionar el primero al cargar (una sola vez).
  useEffect(() => {
    if (!autoSelected.current && models && models.length > 0 && !activeModel) {
      autoSelected.current = true;
      void handleSelect(models[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models]);

  if (isLoading) {
    return <p className="px-1 font-mono text-xs text-fg-subtle">Cargando modelos...</p>;
  }
  if (!models?.length) {
    return <p className="px-1 font-mono text-xs text-fg-subtle">Sin modelos disponibles</p>;
  }

  return (
    <div className="flex flex-col gap-1.5">
      {models.map((m) => (
        <ModelRow
          key={m}
          name={m}
          active={activeModel?.name === m}
          onSelect={() => void handleSelect(m)}
        />
      ))}
    </div>
  );
}
