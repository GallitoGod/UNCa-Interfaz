document.addEventListener('DOMContentLoaded', () => {
  startCamera();
  
})
const { ipcRenderer } = require('electron');

document.getElementById('upload-btn').addEventListener('click', () => {
  ipcRenderer.send('load-image');
});

ipcRenderer.on('image-loaded', (event, { imagePath, imageBuffer }) => {
  console.log('Imagen cargada desde:', imagePath);
});

ipcRenderer.on('image-load-cancelled', () => {
  console.log('La carga de imagen fue cancelada');
});


async function startCamera() {
  const imagePreview = document.getElementById('image-preview');
  const video = document.getElementById('video');

  const constraints = {
    audio: false,
    video: {
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