// AppProviders.tsx — providers globales que envuelven toda la app.
// Por ahora solo TanStack Query; aca se sumarian otros context-providers si hicieran falta.

import { type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5_000,
      retry: 1,
      // Es una app de escritorio local: refetch al enfocar la ventana solo molesta.
      refetchOnWindowFocus: false,
    },
  },
});

export function AppProviders({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
