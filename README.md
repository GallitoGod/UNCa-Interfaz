# Interfaz de Inteligencia Artificial para Procesamiento de Imágenes y Cámaras

Este proyecto es una aplicación de escritorio creada con **Electron**, **JavaScript Vanilla** y **FastApi**, que permite utilizar y cambiar modelos de **inteligencia artificial** para procesar imágenes o capturar datos en tiempo real desde cámaras. Su objetivo principal es proporcionar una interfaz intuitiva para cargar distintos modelos de IA y aplicar análisis sobre imágenes y videos.

## Características:

- **Carga de modelos de IA personalizados**: Permite al usuario cargar diferentes modelos entrenados para tareas específicas, como clasificación de objetos, detección facial, etc.
- **Procesamiento de imágenes**: Los usuarios pueden cargar imágenes locales para ser procesadas por el modelo de IA seleccionado.
- **Integración con cámaras**: Posibilidad de abrir cámaras conectadas al dispositivo y aplicar modelos de IA en tiempo real.
- **Interfaz amigable**: Diseñada con JavaScript Vanilla para ser rápida y fácil de usar, con opciones claras para seleccionar y cambiar entre distintos modelos.
- **Arquitectura flexible**: Ideal para quienes deseen probar distintos modelos de IA sin necesidad de desarrollar interfaces complejas.

## Tecnologías utilizadas:

- **Electron**: Framework para crear aplicaciones de escritorio multiplataforma utilizando tecnologías web.
- **JavaScript Vanilla**: Para la lógica de la aplicación y la interacción con el DOM.
- **FastApi**: Framework para desarrollar APIs en Python.
- **Inteligencia Artificial**: Compatible con modelos preentrenados en formatos populares como `.tflite`, `tf`, `.pt`, `.pth` o `.onnx`.

## Instalación:

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


## Contribución:

- Las contribuciones son bienvenidas. Si deseas agregar nuevas funcionalidades o mejorar el código, no dudes en hacer un pull request.
