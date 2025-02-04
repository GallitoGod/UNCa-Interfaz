import { processFrames } from './frameProcessor.js';
let videoDevices = [];

export async function switchCamera(cameraSelect) {
  try {
    await navigator.mediaDevices.getUserMedia({ video: true });
    const devices = await navigator.mediaDevices.enumerateDevices();
    videoDevices = devices.filter((device) => device.kind === 'videoinput');

    if (videoDevices.length === 0) {
      console.error('No cameras found.');
      return;
    }

    cameraSelect.innerHTML = '';
    videoDevices.forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = device.label || `Camara ${index + 1}`;
      cameraSelect.appendChild(option);
    });

    cameraSelect.addEventListener('change', (event) => {
      const selectedDeviceId = event.target.value;
      startCamera(selectedDeviceId);
    });

    if (videoDevices[0]) {
      startCamera(videoDevices[0].deviceId);
    }
  } catch (err) {
    console.error('Error initializing cameras:', err);
  }
}

async function startCamera(deviceId) {
  if (!deviceId) {
    console.error('Invalid deviceId provided.');
    return;
  }

  const outputCanvas = document.getElementById('outputCanvas');
  const outputCtx = outputCanvas.getContext('2d');
  const imagePreview = document.getElementById('image-preview');
  const video = document.getElementById('video');
  const constraints = {
    audio: false,
    video: {
      deviceId: { exact: deviceId },
      width: imagePreview.clientWidth - 10,
      height: imagePreview.clientHeight - 10,
      frameRate: 30,
    },
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    if (video.srcObject) {
      video.srcObject.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
    }
    video.srcObject = stream;
    await video.play();
    processVideoFrames(video, outputCtx);
    console.log(`Switched to camera: ${deviceId}`);
  } catch (err) {
    console.error(err.name + ': ' + err.message);
  }
}

async function processVideoFrames(video, ctx) {
  const apiEndpoint = "http://127.0.0.1:8000/predict"; 
  const modelType = "tf"; 

  const processFrame = async () => {
    ctx.drawImage(video, 0, 0, ctx.canvas.width, ctx.canvas.height);
    const frameDataURL = ctx.canvas.toDataURL("image/jpeg");

    try {
      const response = await fetch(apiEndpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: frameDataURL,
          model_type: modelType,
        }),
      });

      const result = await response.json();
      drawPrediction(ctx, result.prediction);
    } catch (error) {
      console.error("Error sending frame to API:", error);
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