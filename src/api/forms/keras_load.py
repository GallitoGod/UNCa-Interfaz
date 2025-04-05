import tensorflow as tf

def kerasLoader(model_path):  
    try:
        model = tf.keras.models.load_model(model_path)
        predict_fn = lambda img: model.predict(img).tolist()
        return predict_fn
    except Exception as err:
        return err
    

# 'predict_fn', funciona como una interfaz uniforme de inferencia.