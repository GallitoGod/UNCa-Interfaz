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
            return interpreter.get_tensor(output_details[0]['index']).tolist()
        predict_fn = tflite_predict
        return predict_fn
    except Exception as err:
        return err