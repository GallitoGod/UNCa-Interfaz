// ipc-handlers.js — registro de handlers IPC del MAIN process.
//
// El frontend es un thin client SIN acceso a disco (ver SDD seccion 2 y 4.1.1):
// el listado, la lectura, la escritura y la importacion de modelos/configs ahora
// pasan TODOS por el backend HTTP:
//   - GET  /models             (lista de pesos + hasConfig)
//   - GET  /configs/{name}     (leer config existente)
//   - POST /configs/{name}     (validar + escribir config)
//   - POST /models/upload      (subir pesos por multipart)
// Por eso aca ya NO queda ningun handler de disco (se eliminaron models:list,
// models:import, configs:read y configs:write).
//
// Convencion para IPC futuro NO-disco (si llegara a hacer falta): cada handler debe
// devolver un objeto estandarizado { success: boolean, data?: any, error?: string }
// (SDD 4.1.1) y nunca tirar a traves del puente. Hoy no hay ninguno: la funcion
// queda como no-op para mantener el seam que main.js invoca.

function registerIpcHandlers() {
  // Sin handlers: toda la persistencia vive en el backend (ver nota de arriba).
}

module.exports = { registerIpcHandlers };
