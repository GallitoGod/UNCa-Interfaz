import { streamUrl } from "./constants.js"

export function initVideoStream(videoElement) {
    const ws = new WebSocket(streamUrl);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    let animationFrameId;
  
    ws.onopen = () => {
      console.log('Conexión WebSocket establecida');
      startFrameCapture();
    };
  
    ws.onclose = () => {
      cancelAnimationFrame(animationFrameId);
      console.log('Conexión WebSocket cerrada');
    };
  
    ws.onmessage = (event) => {
      const prediction = JSON.parse(event.data).prediction;
      console.log("Predicción recibida:", prediction);
    };

    function startFrameCapture() {
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
  
      function captureAndSend() {
        if (videoElement.readyState === HTMLMediaElement.HAVE_ENOUGH_DATA) {
          ctx.drawImage(videoElement, 0, 0);
          const frameData = canvas.toDataURL('image/jpeg', 0.8);
          
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(frameData);
          }
        }
        animationFrameId = requestAnimationFrame(captureAndSend);
      }
  
      captureAndSend();
    }
  
    return ws;
}