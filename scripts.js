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
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
   
    const videoElement = document.getElementById('video');
    videoElement.srcObject = stream;
    videoElement.play();

    console.log('Camera started');
  } catch (err) {
    console.error('Error accessing camera.', err);
  }
}