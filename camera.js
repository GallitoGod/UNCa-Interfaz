let videoDevices = []; 
let currentStream = null;

export async function switchCamera(cameraSelect, startCamera) {
  try {
    if (currentStream) {
      currentStream.getTracks().forEach(track => track.stop());
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    videoDevices = devices.filter(device => device.kind === 'videoinput');
    if (videoDevices.length === 0) {
      console.error('No se encontraron camaras disponibles');
      return;
    }

    cameraSelect.innerHTML = '';
    videoDevices.forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = device.label || `Camara ${index + 1}`;
      cameraSelect.appendChild(option);
    });

    startCamera(videoDevices[0].deviceId);

    cameraSelect.addEventListener('change', (event) => {
      const selectedDeviceId = event.target.value;
      startCamera(selectedDeviceId); 
    });

  } catch (err) {
    console.error('Error al inicializar camaras:', err);
  }
}

export async function startCamera(deviceId) {
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
      if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
      }
      currentStream = stream;
      video.srcObject = stream;
      await video.play();
      console.log(`Cambiado a la camara: ${deviceId}`);
    } catch (err) {
      console.log(err.name + ": " + err.message);
    }
  }
  getVideo();
}

