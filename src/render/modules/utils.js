const tf = require('@tensorflow/tfjs');
const ort = require('onnxruntime-web');
import { modelType } from "./constants.js" 

export function preprocessImage(imageData) {
    if (modelType === 'tensorflow') {
        return tf.browser.fromPixels(imageData)
            .resizeNearestNeighbor([224, 224])
            .expandDims(0)
            .toFloat()
            .div(255);
    } else if (modelType === 'onnx') {
        const tensor = tf.browser.fromPixels(imageData)
            .resizeNearestNeighbor([224, 224])
            .expandDims(0)
            .toFloat()
            .div(255);
        const ortTensor = new ort.Tensor('float32', tensor.dataSync(), [1, 3, 224, 224]);
        tensor.dispose();
        return ortTensor;
    }
}