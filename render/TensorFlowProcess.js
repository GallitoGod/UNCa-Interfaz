

export async function tensorFlowModel() {
    const tf = require('@tensorflow/tfjs');
    const model = await tf.loadGraphModel(`models/${modelPath}`);
    const input = tf.tensor(inputData);
    const output = model.predict(input);
    drawOnCanvas(output.dataSync());
}