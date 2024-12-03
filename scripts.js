document.addEventListener('DOMContentLoaded', () => {
  intializeCamera();
  const cameraButtons = {
    mainCamera: document.querySelector('camera-btn'),
    webCamera: document.querySelector('webcam-btn'),
  }
  let videoDevices = [];

  async function intializeCamera () {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      videoDevices = devices.filter(device => device.kind === 'videoinput');

      if (videoDevices.length === 0) {
        console.error('No se encontraron camaras disponibles');
        return;
      }

      console.log('Cámaras disponibles:', videoDevices);

      cameraButtons.mainCamera.addEventListener('click', () => startCamera(0)); 
      cameraButtons.webcam.addEventListener('click', () => startCamera(1)); 
    } catch (err) {
      console.error('Error al inicializar camaras:', err);
    }
  }

  async function switchCamera(cameraIndex) {
    if (cameraIndex >= videoDevices.length || cameraIndex < 0) {
      console.error('Índice de camara invalido');
      return;
    }

    const constraints = {
      video: {
        deviceId: videoDevices[cameraIndex].deviceId
      },
      audio: false
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      video.play();
      console.log(`Cambiado a la cámara: ${videoDevices[cameraIndex].label}`);
    } catch (err) {
      console.error('Error al cambiar de cámara:', err);
    }
  }

});
const { ipcRenderer } = require('electron');



async function startCamera(cameraIndex) {
  const imagePreview = document.getElementById('image-preview');
  const video = document.getElementById('video');
  
  if (cameraIndex >= videoDevices.length || cameraIndex < 0) {
    console.error('Índice de camara invalido');
    return;
  }
  
  const constraints = {
    audio: false,
    video: {
      deviceId: videoDevices[cameraIndex].deviceId,
      width: imagePreview.clientWidth,
      height: imagePreview.clientHeight,
      frameRate: 30
    }
  };

  const getVideo = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      handleSucces(stream);
    } catch (err) {
      console.log(err.name + ": " + err.message);
    }
  }

  const handleSucces = (stream) => {
    video.srcObject = stream;
    video.play();
  }

  getVideo();

  /*
  video.addEventListener('loadeddata', () => {
    imagePreview.style.display = 'block';
  });
  */

}