// detection.service.ts — estrategia de DETECCION (implementada).
// Porta la logica de dibujo del overlay.js viejo al contrato de estrategia.

import type { VisionStrategy, VisionFrameContext } from './types';

// [x1, y1, x2, y2, conf, cls] en px de la imagen original.
type Detection = [number, number, number, number, number, number];

interface DetectionPayload {
  detections?: Detection[];
}

export const detectionStrategy: VisionStrategy<Detection[]> = {
  type: 'detection',
  implemented: true,

  // El WS de deteccion responde { detections: [[x1,y1,x2,y2,conf,cls], ...], error }.
  parse(payload): Detection[] | null {
    const dets = (payload as DetectionPayload | null)?.detections;
    return dets && dets.length > 0 ? dets : null;
  },

  present(detections, { ctx, settings, labelMap }: VisionFrameContext) {
    ctx.lineWidth = 2;
    ctx.font = 'bold 14px sans-serif';
    ctx.textBaseline = 'alphabetic';

    for (const [x1, y1, x2, y2, conf, cls] of detections) {
      ctx.strokeStyle = settings.bboxColor;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const id = Math.round(cls);
      const name = labelMap?.[id] ?? String(id);
      const label = `${name} ${conf.toFixed(2)}`;
      const tw = ctx.measureText(label).width;
      const labelH = 18;
      // Si la caja toca el borde superior, la etiqueta va adentro.
      const ly = y1 >= labelH ? y1 : y1 + labelH;

      ctx.fillStyle = settings.bboxColor;
      ctx.fillRect(x1, ly - labelH, tw + 8, labelH);
      ctx.fillStyle = settings.labelColor;
      ctx.fillText(label, x1 + 4, ly - 5);
    }
  },

  clear() {
    // Deteccion dibuja solo en canvas; el repintado del frame ya lo limpia.
  },
};
