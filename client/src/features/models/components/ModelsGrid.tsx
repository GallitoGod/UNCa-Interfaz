// ModelsGrid.tsx — grilla de modelos (GET /models). Maneja loading / error / vacio.

import { useModelsList } from '../hooks/useModelsList';
import { Spinner } from '@/shared/ui/Spinner';
import { Button } from '@/shared/ui/Button';
import { ModelCard } from './ModelCard';
import type { ApiError } from '@/shared/api/errors';

export function ModelsGrid({
  selectedFile,
  onSelect,
}: {
  selectedFile: string | null;
  onSelect: (file: string) => void;
}) {
  const { data, isLoading, isError, error, refetch } = useModelsList();
  const models = data ?? [];

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-fg">Modelos disponibles</h2>
        <Button variant="ghost" size="sm" onClick={() => void refetch()}>
          Actualizar
        </Button>
      </div>

      {isLoading ? (
        <div className="grid place-items-center py-10">
          <Spinner />
        </div>
      ) : isError ? (
        <p className="text-sm text-fg-subtle">
          No se pudo leer models/ ({(error as ApiError)?.message ?? 'error'})
        </p>
      ) : models.length === 0 ? (
        <p className="text-sm text-fg-subtle">No hay modelos. Arrastra uno al area de la derecha.</p>
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {models.map((entry) => (
            <ModelCard
              key={entry.file}
              entry={entry}
              selected={entry.file === selectedFile}
              onSelect={() => onSelect(entry.file)}
            />
          ))}
        </div>
      )}
    </section>
  );
}
