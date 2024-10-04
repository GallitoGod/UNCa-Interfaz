
const { ipcRenderer } = require('electron');

// Escuchar el clic en el botón para cargar la imagen
document.getElementById('upload-btn').addEventListener('click', () => {
    ipcRenderer.send('load-image');
});

// Escuchar la respuesta cuando la imagen es cargada exitosamente
ipcRenderer.on('image-loaded', (event, { imagePath, imageBuffer }) => {
    console.log('Imagen cargada desde:', imagePath);
    // Aquí puedes usar la imagen en el frontend
    // Puedes por ejemplo, mostrar la imagen en una etiqueta <img> con la ruta imagePath
});

// Manejar si se canceló la carga de imagen
ipcRenderer.on('image-load-cancelled', () => {
    console.log('La carga de imagen fue cancelada');
});