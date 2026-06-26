// vite.config.ts — configuracion del frontend nuevo (React) de UNCaLens.
//
// Vive en la raiz del repo pero su "root" apunta a client/, asi el frontend
// nuevo convive con la app vieja (src/render + static/) SIN tocarla: Vite solo
// mira lo que cuelga de client/index.html. El switch a este build se hace recien
// al final de la migracion (Electron cargara client/dist/index.html).

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';

export default defineConfig({
  // El proyecto React vive en client/ (aislado del backend Python y del Electron main).
  root: path.resolve(__dirname, 'client'),

  plugins: [react(), tailwindcss()],

  resolve: {
    // Alias '@' -> client/src para imports absolutos limpios (ej. '@/features/...').
    alias: { '@': path.resolve(__dirname, 'client/src') },
  },

  // base relativa: imprescindible para que el build funcione bajo file:// en Electron
  // (loadFile produce rutas tipo ./assets/... en vez de /assets/...).
  base: './',

  build: {
    outDir: path.resolve(__dirname, 'client/dist'),
    emptyOutDir: true,
  },

  server: { port: 5173 },
});
