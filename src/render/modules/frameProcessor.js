import { predictUrl } from './constants.js';

export async function processVideoFrames(video, ctx) {
  
    const processFrame = async () => {
      ctx.drawImage(video, 0, 0, ctx.canvas.width, ctx.canvas.height); // <---- Aca esta dibujando antes de tener los datos de la api
      const frameDataURL = ctx.canvas.toDataURL("image/jpeg");
  
      try {
        const response = await fetch(predictUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image: frameDataURL,
          }),
        });
        const result = await response.json();
        drawPrediction(ctx, result.prediction);
      } catch (error) {
        console.error("Error fetching predictions:", error);
      }
      requestAnimationFrame(processFrame);
    };
    requestAnimationFrame(processFrame);
  }
  
  function drawPrediction(ctx, prediction) {
    ctx.fillStyle = "red";
    ctx.font = "20px Arial";
    ctx.fillText(`Prediction: ${prediction}`, 10, 30);
  }