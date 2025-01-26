

export async function onnxModel() {
    const ort = require('onnxruntime-web');
    const session = await ort.InferenceSession.create(`models/${modelPath}`);
    const tensor = new ort.Tensor('float32', inputData, [1, inputData.length]);
    const output = await session.run({ input: tensor });
    drawOnCanvas(output.values().next().value.data);
}