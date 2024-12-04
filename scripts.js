import { switchCamera, startCamera } from './camera.js'; 

document.addEventListener('DOMContentLoaded', () => {
  const cameraSelect = document.getElementById('camera-select');
  const recordButton = document.getElementById('record-btn');
  const video = document.getElementById('video');

  let mediaRecorder = null;
  let recordedChunks = [];
  let isRecording = false;

  switchCamera(cameraSelect, startCamera);

  recordButton.addEventListener('click', () => {
    if (!isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
  });

  function startRecording() {
    if (!video.srcObject) {
      console.error("there is no video stream available to record.");
      return;
    }

    mediaRecorder = new MediaRecorder(video.srcObject, { mimeType: 'video/webm' });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const videoBlob = new Blob(recordedChunks, { type: 'video/webm' });
      recordedChunks = [];

      const videoURL = URL.createObjectURL(videoBlob);

      const a = document.createElement('a');
      a.href = videoURL;
      a.download = 'grabacion.webm';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      console.log('Recording completed and saved.');
    };

    mediaRecorder.start();
    isRecording = true;
    recordButton.textContent = "Detener";
    console.log('Recording started.');
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      isRecording = false;
      recordButton.textContent = "Start Recording";
      console.log('Recording stopped.');
    }
  }
});
