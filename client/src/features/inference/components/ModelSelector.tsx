// ModelSelector.tsx — selecciona el modelo a cargar en el backend y fija el modelo
// activo del workspace (cuyo type enruta la estrategia de presentacion).
// Vive en el slot del Header.

import { useEffect, useRef } from 'react';
import type { ModelType } from '@/shared/api/types';
import { useModels, useSelectModel } from '../hooks/useModels';
import { getModelType } from '../api/models';
import { useWorkspaceStore } from '@/features/vision-workspace/store/workspaceStore';

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

  return (
    <div className="flex items-center gap-2">
      <label className="text-xs font-medium text-fg-muted">Modelo</label>
      <select
        value={activeModel?.name ?? ''}
        disabled={isLoading || !models?.length}
        onChange={(e) => void handleSelect(e.target.value)}
        className="h-8 rounded-[var(--radius-sm)] border border-border bg-control px-2 text-sm text-fg focus-visible:outline-none disabled:opacity-50"
      >
        {!models?.length && <option value="">Sin modelos</option>}
        {models?.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>
    </div>
  );
}
