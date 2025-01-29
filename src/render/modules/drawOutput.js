function drawOutput(ctx, data, classLabels) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const barWidth = ctx.canvas.width / data.length;
    data.forEach((value, index) => {
        const barHeight = value * ctx.canvas.height;
        ctx.fillStyle = 'blue';
        ctx.fillRect(index * barWidth, ctx.canvas.height - barHeight, barWidth, barHeight);

        ctx.fillStyle = 'black';
        ctx.font = '12px Arial';
        ctx.fillText(classLabels[index], index * barWidth, ctx.canvas.height - 5);
    });
}

function drawOutput(ctx, boxes, labels, scores, threshold = 0.5) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    boxes.forEach((box, index) => {
        if (scores[index] >= threshold) {
            const [x, y, width, height] = box;

            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, width, height);

            ctx.fillStyle = 'red';
            ctx.font = '14px Arial';
            ctx.fillText(`${labels[index]} (${scores[index].toFixed(2)})`, x, y - 5);
        }
    });
}

function drawOutput(ctx, mask, classColors) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    const data = imageData.data;

    for (let i = 0; i < mask.length; i++) {
        const classIndex = mask[i];
        const color = classColors[classIndex];

        if (color) {
            data[i * 4] = color[0];     // R
            data[i * 4 + 1] = color[1]; // G
            data[i * 4 + 2] = color[2]; // B
            data[i * 4 + 3] = 128;      // Alpha (semi-transparente)
        }
    }

    ctx.putImageData(imageData, 0, 0);
}