// backend-process.js — ciclo de vida del backend (uvicorn) desde Electron.
//
// Fase 4 tarea 1: el main process arranca uvicorn al iniciar la app y lo mata al
// cerrar, asi el usuario no tiene que levantar el backend a mano. Pensado para
// DESARROLLO (usa el venv del repo). En un empaquetado real habria que apuntar a
// un Python embebido; eso queda fuera de alcance.
//
// Escapes utiles:
//   UNCA_NO_SPAWN=1   -> no arranca uvicorn (cuando ya lo corres a mano en otra terminal)
//   UNCA_PYTHON=<ruta> -> fuerza el interprete de Python a usar

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

let backendProcess = null;

// Resuelve el interprete de Python: UNCA_PYTHON > venv del repo > python del PATH.
function resolvePython(projectRoot) {
  if (process.env.UNCA_PYTHON) return process.env.UNCA_PYTHON;

  const venvWin = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
  const venvNix = path.join(projectRoot, '.venv', 'bin', 'python');

  if (process.platform === 'win32' && fs.existsSync(venvWin)) return venvWin;
  if (fs.existsSync(venvNix)) return venvNix;

  // Fallback: lo que haya en el PATH.
  return process.platform === 'win32' ? 'python' : 'python3';
}

// Arranca uvicorn (idempotente: si ya hay un proceso vivo, no lanza otro).
function startBackend({
  projectRoot,
  host = '127.0.0.1',
  port = 8000,
  logger = console,
} = {}) {
  if (process.env.UNCA_NO_SPAWN === '1') {
    logger.log(
      '[backend] UNCA_NO_SPAWN=1: no se arranca uvicorn (modo manual).'
    );
    return null;
  }
  if (backendProcess) return backendProcess;

  const python = resolvePython(projectRoot);
  const args = [
    '-m',
    'uvicorn',
    'api.mainAPI:app',
    '--host',
    host,
    '--port',
    String(port),
    '--app-dir',
    'src',
  ];

  logger.log(
    `[backend] iniciando: ${python} ${args.join(' ')} (cwd=${projectRoot})`
  );

  backendProcess = spawn(python, args, {
    cwd: projectRoot,
    stdio: 'pipe',
    windowsHide: true,
  });

  backendProcess.stdout.on('data', (d) =>
    logger.log(`[backend] ${d.toString().trimEnd()}`)
  );
  backendProcess.stderr.on('data', (d) =>
    logger.log(`[backend] ${d.toString().trimEnd()}`)
  );
  backendProcess.on('exit', (code, signal) => {
    logger.log(`[backend] termino (code=${code} signal=${signal}).`);
    backendProcess = null;
  });
  backendProcess.on('error', (err) => {
    logger.error(`[backend] no se pudo iniciar: ${err.message}`);
    backendProcess = null;
  });

  return backendProcess;
}

// Detiene uvicorn. En Windows mata el arbol entero (uvicorn puede tener hijos);
// en POSIX manda SIGTERM al proceso.
function stopBackend({ logger = console } = {}) {
  if (!backendProcess) return;
  const proc = backendProcess;
  backendProcess = null;

  try {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', String(proc.pid), '/T', '/F'], {
        windowsHide: true,
      });
    } else {
      proc.kill('SIGTERM');
    }
    if (logger) logger.log('[backend] detenido.');
  } catch (e) {
    if (logger) logger.error(`[backend] error al detener: ${e.message}`);
  }
}

module.exports = { startBackend, stopBackend, resolvePython };
