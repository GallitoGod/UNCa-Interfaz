// App.tsx — raiz del arbol React. Providers + chrome (Header) + ViewRouter.

import { AppProviders } from './providers/AppProviders';
import { Header } from './components/Header';
import { ViewRouter } from './router';

export function App() {
  return (
    <AppProviders>
      <div className="flex min-h-screen flex-col bg-canvas text-fg">
        {/* Title bar: marca, navegacion y modelo activo (la seleccion vive en el
            panel izquierdo de Inferencia). */}
        <Header />
        {/* min-h-0: deja que el contenido scrollee/encoja dentro del flex column */}
        <main className="min-h-0 flex-1">
          <ViewRouter />
        </main>
      </div>
    </AppProviders>
  );
}
