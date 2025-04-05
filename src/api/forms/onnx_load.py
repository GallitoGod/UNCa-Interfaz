import onnxruntime as ort

def onnxLoader(model_path):
    try:    
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        predict_fn = lambda img: session.run(None, {input_name: img})[0].tolist()
        return predict_fn
    except Exception as err:
        return err