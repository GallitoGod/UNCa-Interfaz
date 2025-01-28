const tf = window.tf;
const ort = window.ort;

let selectedModel = null;
let isProcessing = false;
let modelType = null; // 'tensorflow' o 'onnx'

export async function getModels() {
    const models = await globalThis.api.getModels();
    const modelSelect = document.getElementById('ai-model');

    modelSelect.innerHTML = '';
    models.forEach((model) => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
    }) 

    modelSelect.addEventListener('change',async () => {
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
    });
}


async function runModel(inputTensor) {
  if (!selectedModel || !modelType) {
    throw new Error('Modelo no cargado o tipo desconocido');
  }

  if (modelType === 'tensorflow') {
    const output = selectedModel.predict(inputTensor);
    const result = output.dataSync();
    output.dispose(); 
    return result;
  } else if (modelType === 'onnx') {
    const inputName = selectedModel.inputNames[0]; 
    const feeds = { [inputName]: new ort.Tensor('float32', inputTensor.dataSync(), inputTensor.shape) };
    const results = await selectedModel.run(feeds);
    const outputName = selectedModel.outputNames[0]; 
    return results[outputName].data;
  }
}

function preprocessImage(imageData) {

  if (modelType === 'tensorflow') {
    return tf.browser.fromPixels(imageData).resizeNearestNeighbor([224, 224]).expandDims(0).toFloat().div(255);
  } else if (modelType === 'onnx') {
    const ortTensor = new ort.Tensor('float32', tfTensor.dataSync(), [1, 3, 224, 224]);
    return ortTensor;
  }
}

export async function processFrames(video, ctx) {
  const tempCanvas = document.createElement('canvas');
  const tempCtx = tempCanvas.getContext('2d');

  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;

  async function processFrame() {
    if (isProcessing || !selectedModel) {
      requestAnimationFrame(processFrame);
      return;
    }
  
    isProcessing = true;
  
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const inputTensor = preprocessImage(imageData);
    const prediction = await runModel(inputTensor);
    drawOutput(ctx, prediction);

    isProcessing = false;
    requestAnimationFrame(processFrame);
  }
  requestAnimationFrame(processFrame);
}
  
function drawOutput(ctx, data) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  const barWidth = ctx.canvas.width / data.length;
  data.forEach((value, index) => {
    const barHeight = value * ctx.canvas.height;
    ctx.fillStyle = 'blue';
    ctx.fillRect(index * barWidth, ctx.canvas.height - barHeight, barWidth, barHeight);
  });
}