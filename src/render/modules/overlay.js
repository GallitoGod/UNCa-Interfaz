// Dibujo de detecciones sobre el canvas de salida (client-side).
// El backend ya no dibuja: devuelve [x1, y1, x2, y2, conf, cls] en pixeles
// de la imagen original y este modulo las superpone con los colores elegidos.

export const drawSettings = {
  bboxColor: '#00BFFF',
  labelColor: '#000000',
};

export function drawDetections(ctx, detections) {
  ctx.lineWidth = 2;
  ctx.font = 'bold 14px sans-serif';
  ctx.textBaseline = 'alphabetic';

  for (const [x1, y1, x2, y2, conf, cls] of detections) {
    ctx.strokeStyle = drawSettings.bboxColor;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${Math.round(cls)} ${conf.toFixed(2)}`;
    const tw = ctx.measureText(label).width;
    const labelH = 18;
    // si la caja toca el borde superior, la etiqueta va adentro
    const ly = y1 >= labelH ? y1 : y1 + labelH;

    ctx.fillStyle = drawSettings.bboxColor;
    ctx.fillRect(x1, ly - labelH, tw + 8, labelH);
    ctx.fillStyle = drawSettings.labelColor;
    ctx.fillText(label, x1 + 4, ly - 5);
  }
}
