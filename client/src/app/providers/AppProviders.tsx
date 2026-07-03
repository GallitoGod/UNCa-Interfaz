// AppProviders.tsx — providers globales que envuelven toda la app.
// Por ahora solo TanStack Query; aca se sumarian otros context-providers si hicieran falta.

import { type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiError } from '@/shared/api/errors';

// Electron arranca uvicorn en paralelo a la ventana (backend-process.js), asi que
// las primeras queries pueden llegar ANTES de que el backend termine de bootear
// (los imports de TF tardan varios segundos). El interceptor de axios normaliza
// todo a ApiError: kind 'network' (sin respuesta) o 'timeout' significan "backend
// todavia no arriba" y se reintenta con paciencia. Un error HTTP real (4xx/5xx)
// es una respuesta del backend y se reintenta una sola vez.
function isBackendDown(error: unknown): boolean {
  return error instanceof ApiError && (error.kind === 'network' || error.kind === 'timeout');
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5_000,
      retry: (failureCount, error) => (isBackendDown(error) ? failureCount < 20 : failureCount < 1),
      // Backoff corto y acotado: cubre ~30s de boot sin martillar el puerto.
      retryDelay: (attempt) => Math.min(1_000 * 2 ** attempt, 3_000),
      // Es una app de escritorio local: refetch al enfocar la ventana solo molesta.
      refetchOnWindowFocus: false,
    },
  },
});

export function AppProviders({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
