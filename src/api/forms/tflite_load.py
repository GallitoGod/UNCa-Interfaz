import tensorflow as tf

def tfliteLoader(model_path):
    try:    
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        def tflite_predict(img):
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            outs = tuple(interpreter.get_tensor(od['index']) for od in output_details)
            return outs[0] if len(outs) == 1 else outs
        predict_fn = tflite_predict
        return predict_fn
    except Exception as err:
        return err