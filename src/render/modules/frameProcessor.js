import { predictUrl } from './constants.js';

/*
  Recibir imagenes procesadas al cliente:
  Ya estoy recibiendo una imagen, pero viene con parametros para procesarla en el cliente, eso esta mal
solo necesito recibir una imagen en base64 ya dibujada con todo lo necesario para, en el cliente, unicamente dibujarla 
en un <canvas> o un <img> 

  IDEAS:
    - JS tiene un objeto llamado Image, puede ser utli para actualizar mas rapido entre frames, algo como:
      let img = new Image();
      img.src = "data:image/jpeg;base64," + processedImage;
      ctx.drawImage(img, 0, 0);
    (En este caso, si uso <img>, simplemente pongo el src del Image y ya esta todo)
*/

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