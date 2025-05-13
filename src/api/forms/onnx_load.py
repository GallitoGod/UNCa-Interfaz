from generalModel import GeneralModel
import onnxruntime as ort

class OnnxInterpreter(GeneralModel):

    @staticmethod
    def loader(model_path):
        try:    
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            predict_fn = lambda img: session.run(None, {input_name: img})[0].tolist()
            return predict_fn
        except Exception as err:
            return err

    @staticmethod
    def input_adapter(input_data):
        return super().input_adapter(input_data)    #<---- TODAVIA TENGO QUE LEER COMO UTILIZA LOS INPUTS ONNX
    
    @staticmethod
    def output_adapter(inference_data):
        return super().output_adapter(inference_data)   #<---- TODAVIA TENGO QUE LEER COMO UTILIZA LOS OUTPUTS ONNX



# def onnxLoader(model_path):
#     try:    
#         session = ort.InferenceSession(model_path)
#         input_name = session.get_inputs()[0].name
#         predict_fn = lambda img: session.run(None, {input_name: img})[0].tolist()
#         return predict_fn
#     except Exception as err:
#         return err