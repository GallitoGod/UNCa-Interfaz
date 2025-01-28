import * as tf from '@tensorflow/tfjs';
import * as tflite from 'tflite';
import * as ort from 'onnxruntime-web';
import { selectedModel, modelType } from './constants.js';

/**
 * Ejecuta el modelo de IA con el tensor de entrada y devuelve la salida procesada.
 * @param {tf.Tensor|ort.Tensor} inputTensor - Tensor de entrada (imagen preprocesada).
 * @returns {Promise<Array|Object>} - Resultados de la inferencia (depende del modelo).
 */
export async function runModel(inputTensor) {
    if (!selectedModel || !modelType) {
        throw new Error('Modelo no cargado o tipo desconocido');
    }

    try {
        if (modelType === 'tensorflow') {
            return await runTensorFlowModel(inputTensor);
        } else if (modelType === 'onnx') {
            return await runONNXModel(inputTensor);
        }// else if (modelType === 'tflite') {
        //    return await runTFLiteModel(inputTensor);
        //} 
        else {
            throw new Error('Tipo de modelo no soportado');
        }
    } catch (error) {
        console.error('Error al ejecutar el modelo:', error);
        throw error;
    } finally {
        if (modelType === 'tensorflow') {
            inputTensor.dispose();
        }
    }
}

/**
 * Ejecuta un modelo de TensorFlow.js.
 * @param {tf.Tensor} inputTensor - Tensor de entrada.
 * @returns {Array} - Resultados de la inferencia.
 */
async function runTensorFlowModel(inputTensor) {
    const output = selectedModel.predict(inputTensor);
    const result = output.dataSync(); // Convertir a array de JavaScript
    output.dispose(); // Liberar memoria del tensor de salida
    return result;
}

/**
 * Ejecuta un modelo de ONNX.
 * @param {ort.Tensor} inputTensor - Tensor de entrada.
 * @returns {Array} - Resultados de la inferencia.
 */
async function runONNXModel(inputTensor) {
    const inputName = selectedModel.inputNames[0];
    const feeds = { [inputName]: inputTensor };
    const results = await selectedModel.run(feeds);
    const outputName = selectedModel.outputNames[0];
    return results[outputName].data;
}

/**
 * Ejecuta un modelo de TFLite (TensorFlow Lite).
 * @param {tf.Tensor} inputTensor - Tensor de entrada.
 * @returns {Array} - Resultados de la inferencia.
 */
// async function runTFLiteModel(inputTensor) {
//     const tfliteModel = await tflite.loadModel('models/model.tflite');
//     const output = tfliteModel.predict(inputTensor);
//     return output.dataSync();
// }