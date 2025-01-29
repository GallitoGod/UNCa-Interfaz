const tf = require('@tensorflow/tfjs');
const ort = require('onnxruntime-web');
import { selectedModel, modelType } from "./constants.js";
import { sectionIA } from "./uiManager.js";

export async function setupModelLoader() {
    const modelSelect = document.getElementById('ai-model');
    const models = await globalThis.api.getModels();
    sectionIA(models, modelSelect);

    modelSelect.addEventListener('change', async () => {
        try {
            const modelPath = modelSelect.value;
            if (modelPath.endsWith('.onnx')) {
                selectedModel = await ort.InferenceSession.create(`models/${modelPath}`);
                modelType = 'onnx';
                console.log('Modelo ONNX cargado:', modelPath);
            } else if (modelPath.endsWith('.json')) {
                selectedModel = await tf.loadGraphModel(`models/${modelPath}`);
                modelType = 'tensorflow';
                console.log('Modelo TensorFlow.js cargado:', modelPath);
            } else {
                console.error('Formato de modelo no soportado:', modelPath);
                selectedModel = null;
                modelType = null;
            }
        } catch (error) {
            console.error('Error al cargar el modelo:', error);
            selectedModel = null;
            modelType = null;
        }
    });
}