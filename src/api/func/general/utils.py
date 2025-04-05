#   ---->   PRUEBA DE FINDCONFIG  <----
def findConfigModel(model_path):
    import os
    print(model_path)
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    print(base_name)
    config_path = os.path.join("configs", base_name + ".json")
    print(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró config para {base_name}")
    
    

findConfigModel("/models/yolov7-tiny.onnx")