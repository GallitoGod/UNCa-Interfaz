import tensorflow as tf
import numpy as np

def tfliteLoader(model_path):

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_index = input_details[0]['index']
    in_shape = tuple(input_details[0]['shape'])     
    in_dtype = input_details[0]['dtype']            

    def tflite_predict(img):
        arr = np.asarray(img)

        if arr.shape == in_shape[1:]:
            arr = arr[None, ...]  # np.expand_dims(arr, 0)

        if tuple(arr.shape) != in_shape:
            raise ValueError(f"TFLite: shape de entrada {arr.shape} != esperada {in_shape}")

        if arr.dtype != in_dtype:
            arr = arr.astype(in_dtype, copy=False)

        interpreter.set_tensor(in_index, arr)
        interpreter.invoke()

        outs = tuple(interpreter.get_tensor(od['index']) for od in output_details)
        return outs[0] if len(outs) == 1 else outs

    return tflite_predict
