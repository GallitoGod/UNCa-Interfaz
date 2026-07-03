/// <reference types="vite/client" />

// Variables de entorno del frontend (Vite las inyecta desde .env / import.meta.env).
interface ImportMetaEnv {
  readonly VITE_API_URL?: string; // ej. http://127.0.0.1:8000
  readonly VITE_WS_URL?: string; // ej. ws://127.0.0.1:8000
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
