from generalModel import GeneralModel
import tensorflow as tf


class KerasInterpreter(GeneralModel):

    @staticmethod
    def loader(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            predict_fn = lambda img: model.predict(img).tolist()
            return predict_fn
        except Exception as err:
            return err

    @staticmethod
    def input_adapter(input_data):
        return super().input_adapter(input_data)    #<---- TODAVIA TENGO QUE LEER COMO UTILIZA LOS INPUTS KERAS


    @staticmethod
    def output_adapter(inference_data):
        return super().output_adapter(inference_data)   #<---- TODAVIA TENGO QUE LEER COMO UTILIZA LOS OUTPUTS KERAS


# def kerasLoader(model_path):  
#     try:
#         model = tf.keras.models.load_model(model_path)
#         predict_fn = lambda img: model.predict(img).tolist()
#         return predict_fn
#     except Exception as err:
#         return err
    
