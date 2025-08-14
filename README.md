# UNCaLens – Sistema Modular de Visión por Computadora

Este proyecto corresponde a una aplicación diseñada para la ejecución y prueba de modelos de detección de objetos sobre imágenes y videos. Actualmente está en desarrollo y enfocado principalmente en la parte backend, con un pipeline de inferencia funcional y adaptable a distintos formatos de modelos.

> [!IMPORTANT]
> Estado del Proyecto
>- Actualmente, la comunicación entre el backend (Python/FastAPI) y el cliente en JavaScript/Electron se encuentra en desarrollo.
>- El sistema ya soporta la carga y ejecución de modelos de detección en formatos Keras, ONNX y TFLite.
>- El frontend en Electron aún no está operativo; las pruebas actuales se realizan directamente sobre la API y scripts de backend.

## Características Principales
- Compatibilidad con múltiples formatos de modelos: `ONNX`, `Keras (.h5)` y `TFLite`.
- Procesamiento de imágenes y video:
   - Escalado y preprocesado automático a las dimensiones requeridas por cada modelo.
   - Postprocesamiento con filtrado por confianza, NMS y reversión de transformaciones (letterbox).
- Diseño modular: Separación clara entre el manejo del modelo, preprocesamiento, inferencia y postprocesamiento.
- Preparado para integración con cámaras y flujos de video en tiempo real (funcionalidad planificada).

## Tecnologías Utilizadas
- `Python 3.8.10` – Lenguaje principal para backend.
- `FastAPI` – Servidor API rápido y asíncrono.
- `OpenCV` – Procesamiento de imágenes y video.
- `NumPy` – Manipulación numérica.
- `ONNX Runtime` / `TensorFlow Lite` / `TensorFlow` – Ejecución de modelos.


> [!NOTE]
> Instalación:

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/GallitoGod/UNCa-Interfaz.git
   cd UNCa-Interfaz
   ```
2. Instalar las dependencias de Node.js:
   ```bash
   npm install
   npm install electron --save-dev
   ```
3. Configurar el entorno de Python:

   - **Es necesario el interprete de python 3.8.10**

   - Windows:
   ```bash
   python -m venv venv
   .venv\Scripts\activate
   ```
   - Linux/MacOS:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. Instalar las dependencias de Python:
   ```bash
   pip install -r requirements.txt
   ```
5. Ejecutar la aplicación:
   - En una terminal, inicia el servidor de Python:
   ```bash
   fastapi dev src/TensorAPI.py
   ```
   - En otra terminal, inicia el cliente de Electron:
   ```bash
   npm start
   ```

> [!WARNING]
> Nota: El cliente en Electron todavía no está operativo. Las pruebas deben realizarse consumiendo directamente la API.


## Contribución
El proyecto está en una etapa temprana, por lo que cualquier aporte es bienvenido.
Si querés colaborar, podés:
- Probar la API y reportar errores.
- Mejorar el pipeline de inferencia y postprocesamiento.
- Avanzar en la integración con el cliente en Electron.