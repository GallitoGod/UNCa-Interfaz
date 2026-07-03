// useModels.ts — query de modelos disponibles + mutation de seleccion.

import { useMutation, useQuery } from '@tanstack/react-query';
import { getModels, selectModel } from '../api/models';

export function useModels() {
  return useQuery({ queryKey: ['models'], queryFn: getModels });
}

export function useSelectModel() {
  return useMutation({ mutationFn: selectModel });
}
