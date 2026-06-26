// main.tsx — bootstrap del renderer React.
// Monta <App/> en #root. El wiring real (providers, router) vive dentro de App.

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from '@/app/App';
import './index.css';

const rootEl = document.getElementById('root');
if (!rootEl) {
  // Falla ruidosa: si no existe #root algo se rompio en index.html.
  throw new Error('No se encontro el elemento #root para montar React');
}

createRoot(rootEl).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
