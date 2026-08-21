// classification.service.ts — estrategia de CLASIFICACION (implementada 2026-08-13).
//
// A diferencia de deteccion, clasificacion NO dibuja en el canvas: el resultado no
// tiene geometria, es texto. Se presenta como un panel HTML en overlayRoot (la capa
// que VisionWorkspace monta encima del canvas).
//
// Contrato del backend (tasks/classification.py -> serialize_classification):
//   { task: 'classification', result: [{ cls: number, score: number }, ...], error }
// El result ya viene filtrado por umbral, cortado a top_k y ordenado por score desc.

import type { VisionStrategy, VisionFrameContext } from './types';

export interface ClassPrediction {
  cls: number;
  score: number;
}

interface ClassificationPayload {
  task?: string;
  result?: ClassPrediction[];
}

// Nodos del panel, cacheados POR overlayRoot. Reconstruir el DOM en cada frame a
// 30 fps seria un desperdicio: se crean una vez y despues solo se actualiza el
// texto. WeakMap para no retener el overlay si el workspace se desmonta.
interface PanelNodes {
  root: HTMLDivElement;
  list: HTMLDivElement;
  rows: { row: HTMLDivElement; name: HTMLSpanElement; score: HTMLSpanElement }[];
}
const panels = new WeakMap<HTMLElement, PanelNodes>();

function createPanel(overlayRoot: HTMLElement): PanelNodes {
  const root = document.createElement('div');
  // Estilos inline con los tokens --c-* de index.css: no dependen de que Tailwind
  // escanee este archivo, y siguen la piel Cabina Tecnica.
  // Esquina INFERIOR izquierda, no la superior: arriba-izquierda la ocupa el
  // MetricsHUD (left-3 top-3, mismas coordenadas exactas) y arriba-derecha el badge
  // "En vivo" de camara. Verificado el 2026-08-21: con top:12px el chip de FPS tapaba
  // el titulo del panel y la primera clase.
  root.style.cssText = [
    'position:absolute',
    'left:12px',
    'bottom:12px',
    'min-width:150px',
    'padding:8px 10px',
    'border-radius:var(--radius-md)',
    'border:1px solid var(--c-accent-border)',
    'background:rgba(10,13,19,.82)',
    'backdrop-filter:blur(6px)',
    'pointer-events:none',
  ].join(';');

  const title = document.createElement('div');
  title.textContent = 'Clasificacion';
  title.style.cssText = [
    "font-family:'JetBrains Mono',monospace",
    'font-size:9px',
    'letter-spacing:.12em',
    'text-transform:uppercase',
    'color:var(--c-label)',
    'margin-bottom:6px',
  ].join(';');

  const list = document.createElement('div');
  list.style.cssText = 'display:flex;flex-direction:column;gap:3px';

  root.appendChild(title);
  root.appendChild(list);
  overlayRoot.appendChild(root);

  const nodes: PanelNodes = { root, list, rows: [] };
  panels.set(overlayRoot, nodes);
  return nodes;
}

function getPanel(overlayRoot: HTMLElement): PanelNodes {
  const cached = panels.get(overlayRoot);
  // Revalidar el parentesco: si el overlay se limpio por fuera, el nodo cacheado
  // quedo huerfano y hay que rehacerlo.
  if (cached && cached.root.parentElement === overlayRoot) return cached;
  return createPanel(overlayRoot);
}

// Ajusta la cantidad de filas del panel a 'n', reutilizando las que ya existen.
function ensureRows(panel: PanelNodes, n: number): void {
  while (panel.rows.length < n) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:baseline;gap:10px;justify-content:space-between';

    const name = document.createElement('span');
    name.style.cssText = "font-size:12px;color:var(--c-fg);font-family:'Space Grotesk',sans-serif";

    const score = document.createElement('span');
    score.style.cssText = [
      "font-family:'JetBrains Mono',monospace",
      'font-size:12px',
      'font-variant-numeric:tabular-nums',
      'color:var(--c-accent)',
    ].join(';');

    row.appendChild(name);
    row.appendChild(score);
    panel.list.appendChild(row);
    panel.rows.push({ row, name, score });
  }
  while (panel.rows.length > n) {
    const extra = panel.rows.pop();
    extra?.row.remove();
  }
}

export const classificationStrategy: VisionStrategy<ClassPrediction[]> = {
  type: 'classification',
  implemented: true,

  // Devuelve [] (no null) cuando no hay clases sobre el umbral: null significaria
  // "no presentes nada" y el panel del frame anterior quedaria congelado en pantalla.
  // Con [] llegamos a present() y el panel se oculta, que es lo correcto.
  parse(payload): ClassPrediction[] | null {
    const p = payload as ClassificationPayload | null;
    if (p?.task && p.task !== 'classification') return null;
    return Array.isArray(p?.result) ? (p.result as ClassPrediction[]) : null;
  },

  present(predictions, { overlayRoot, labelMap }: VisionFrameContext) {
    const panel = getPanel(overlayRoot);

    // Sin clases sobre el umbral: se oculta el panel en vez de mostrarlo vacio.
    if (predictions.length === 0) {
      panel.root.style.display = 'none';
      return;
    }
    panel.root.style.display = '';

    ensureRows(panel, predictions.length);
    predictions.forEach((pred, i) => {
      const row = panel.rows[i];
      // Sin label_map en el config, el backend solo puede dar el id numerico.
      row.name.textContent = labelMap?.[pred.cls] ?? `clase ${pred.cls}`;
      row.score.textContent = pred.score.toFixed(2);
    });
  },

  clear({ overlayRoot }: VisionFrameContext) {
    const panel = panels.get(overlayRoot);
    if (!panel) return;
    panel.root.remove();
    panels.delete(overlayRoot);
  },
};
