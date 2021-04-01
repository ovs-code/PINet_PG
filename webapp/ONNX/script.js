// This code will be executed in the beginning of main.js
const keypointSess = new onnx.InferenceSession();
const segmentationSess = new onnx.InferenceSession();
const transferSess = new onnx.InferenceSession();
const loadingKeypointModelPromise = keypointSess.loadModel('modelpath')
const loadingSegementationModelPromise = segmentationSess.loadModel('modelpath')
const loadingTransferModelPromise = transferSess.loadModel('modelpath')

// function for doing the actual predictions
// this function will be called on submit of all the form data
async function generateTransferedPose() {
    const inputImage = null;
    const inputPose = null;

    // convert the input from the formdata to Tensors
    const imageTensor = new onnx.Tensor(new Float32Array(inputImage.data), 'float32');
    const poseTensor = new onnx.Tensor(new Float32Array(inputPose.data), 'float32');

    // wait for the models to be ready
    await loadingKeypointModelPromise;
    const outputKeypointMap = await keypointSess.run([imageTensor]);
    const outputKeypointTensor = outputKeypointMap.values().next().value;

    await loadingSegementationModelPromise;
    const outputKeypointMap = await keypointSess.run([imageTensor]);
    const outputKeypointTensor = outputKeypointMap.values().next().value;
    await loadingTransferModelPromise;
    const 
}