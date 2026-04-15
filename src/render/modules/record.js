let mediaRecorder = null;
let recordedChunks = [];
export let isRecording = false;

export function startRecording(recordButton, video) {
  if (!video.srcObject) {
    console.error('No hay stream de video disponible para grabar.');
    return;
  }

  mediaRecorder = new MediaRecorder(video.srcObject, {
    mimeType: 'video/webm',
  });

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
    console.log('Grabacion completada y guardada.');
  };

  mediaRecorder.start();
  isRecording = true;
  recordButton.textContent = 'Detener';
  recordButton.style.backgroundColor = '#ff4136';
  console.log('Grabacion iniciada.');
}

export function stopRecording(recordButton) {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    recordButton.textContent = 'Iniciar';
    recordButton.style.backgroundColor = '#3b82f6';
    console.log('Grabacion detenida.');
  }
}
