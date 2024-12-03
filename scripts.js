document.addEventListener('DOMContentLoaded', () => {
  const cameraSelect = document.getElementById('camera-select');

  let videoDevices = []; 

  async function initializeCameras() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      videoDevices = devices.filter(device => device.kind === 'videoinput');

      if (videoDevices.length === 0) {
        console.error('No se encontraron cámaras disponibles');
        return;
      }

      cameraSelect.innerHTML = '';
      videoDevices.forEach((device, index) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.textContent = device.label || `Cámara ${index + 1}`;
        cameraSelect.appendChild(option);
      });

      startCamera(videoDevices[0].deviceId);
    } catch (err) {
      console.error('Error al inicializar cámaras:', err);
    }
  }

});


const { ipcRenderer } = require('electron');



async function startCamera(cameraIndex) {
  const imagePreview = document.getElementById('image-preview');
  const video = document.getElementById('video');
  
  const constraints = {
    audio: false,
    video: {
      deviceId: { exact: deviceId },
      width: imagePreview.clientWidth,
      height: imagePreview.clientHeight,
      frameRate: 30
    }
  };

  const getVideo = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      handleSucces(stream);
      console.log(`Cambiado a la cámara: ${videoDevices[cameraIndex].label}`);
    } catch (err) {
      console.log(err.name + ": " + err.message);
    }
  }

  const handleSucces = (stream) => {
    video.srcObject = stream;
    video.play();
  }

  getVideo();

  // Escucha cambios en el menú desplegable
  cameraSelect.addEventListener('change', (event) => {
    switchCamera(event.target.value);
  });

  // Inicializa las cámaras al cargar la página
  initializeCameras();

}