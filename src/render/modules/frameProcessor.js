import { selectedModel, modelType } from './constants.js';
import { preprocessImage } from './utils.js';
import { runModel } from './modelAction.js';

export function setupFrameProcessor() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('output-canvas');
    const ctx = canvas.getContext('2d');

    let isProcessing = false;

    async function processFrame() {
        if (isProcessing || !selectedModel) return;
        isProcessing = true;

        try {
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            tempCanvas.width = video.videoWidth;
            tempCanvas.height = video.videoHeight;

            tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
            const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
            const inputTensor = preprocessImage(imageData);
            const prediction = await runModel(inputTensor);
            drawOutput(ctx, prediction);
        } catch (error) {
            console.error('Error al procesar el frame:', error);
        } finally {
            isProcessing = false;
            requestAnimationFrame(processFrame);
        }
    }

    requestAnimationFrame(processFrame);
}

function drawOutput(ctx, data) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const barWidth = ctx.canvas.width / data.length;
    data.forEach((value, index) => {
        const barHeight = value * ctx.canvas.height;
        ctx.fillStyle = 'blue';
        ctx.fillRect(index * barWidth, ctx.canvas.height - barHeight, barWidth, barHeight);
    });
}