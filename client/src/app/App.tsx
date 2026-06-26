// App.tsx — raiz del arbol React. Providers + chrome (Header) + ViewRouter.

import { AppProviders } from './providers/AppProviders';
import { Header } from './components/Header';
import { ViewRouter } from './router';
import { ModelSelector } from '@/features/inference/components/ModelSelector';

export function App() {
  return (
    <AppProviders>
      <div className="flex min-h-screen flex-col bg-canvas text-fg">
        {/* El selector de modelo vive siempre en el header (slot), como en la app vieja. */}
        <Header modelSelector={<ModelSelector />} />
        {/* min-h-0: deja que el contenido scrollee/encoja dentro del flex column */}
        <main className="min-h-0 flex-1">
          <ViewRouter />
        </main>
      </div>
    </AppProviders>
  );
}
