# import torch
# import torch.nn as nn
# import os



# class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
#     def _check_input_dim(self, input):
#         # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
#         # is this method that is overwritten by the sub-class
#         # This original goal of this method was for tensor sanity checks
#         # If you're ok bypassing those sanity checks (eg. if you trust your inference
#         # to provide the right dimensional inputs), then you can just use this method
#         # for easy conversion from SyncBatchNorm
#         # (unfortunately, SyncBatchNorm does not store the original class - if it did
#         #  we could return the one that was originally created)
#         return

# def revert_sync_batchnorm(module):
#     # this is very similar to the function that it is trying to revert:
#     # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
#     module_output = module
#     if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
#         new_cls = BatchNormXd
#         module_output = BatchNormXd(module.num_features,
#                                                module.eps, module.momentum,
#                                                module.affine,
#                                                module.track_running_stats)
#         if module.affine:
#             with torch.no_grad():
#                 module_output.weight = module.weight
#                 module_output.bias = module.bias
#         module_output.running_mean = module.running_mean
#         module_output.running_var = module.running_var
#         module_output.num_batches_tracked = module.num_batches_tracked
#         if hasattr(module, "qconfig"):
#             module_output.qconfig = module.qconfig
#     for name, child in module.named_children():
#         module_output.add_module(name, revert_sync_batchnorm(child))
#     del module
#     return module_output

# class TracedModel(nn.Module):

#     def __init__(self, model=None, device=None, img_size=(640,640)): 
#         super(TracedModel, self).__init__()
        
#         print(" Convert model to Traced-model... ") 
#         self.stride = model.stride
#         self.names = model.names
#         self.model = model

#         self.model = revert_sync_batchnorm(self.model)
#         self.model.to('cpu')
#         self.model.eval()

#         self.detect_layer = self.model.model[-1]
#         self.model.traced = True
        
#         rand_example = torch.rand(1, 3, img_size, img_size)
        
#         traced_script_module = torch.jit.trace(self.model, rand_example, strict=False)
#         #traced_script_module = torch.jit.script(self.model)
#         traced_script_module.save("traced_model.pt")
#         print(" traced_script_module saved! ")
#         self.model = traced_script_module
#         self.model.to(device)
#         self.detect_layer.to(device)
#         print(" model is traced! \n") 

#     def forward(self, x, augment=False, profile=False):
#         out = self.model(x)
#         out = self.detect_layer(out)
#         return out

# ruta = os.path.abspath("models\yolov7.pt")
# print("Esta es la ruta: -----> ",ruta)
# model = torch.load(ruta, map_location="cpu")
# model.eval()
# traced_model = TracedModel(model, device='cuda' if torch.cuda.is_available() else 'cpu')

# print(model)  # Busca la primera capa para ver la entrada esperada


import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import random
import json
import os

#  Cargar cualquier modelo ONNX
def load_model(onnx_path, class_names_path=None, use_cuda=False):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    #  Cargar clases desde JSON si existe
    if class_names_path and os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
    else:
        class_names = [f"class_{i}" for i in range(100)]  # Si no hay clases, usar nombres genéricos

    #  Generar colores aleatorios para cada clase
    colors = {name: [random.randint(0, 255) for _ in range(3)] for name in class_names}

    return session, class_names, colors

#  Preprocesamiento de imágenes
def preprocess(img_path, img_size=(640, 640)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def letterbox(im, new_shape):
        shape = im.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return im, r, (dw, dh)

    image, ratio, dwdh = letterbox(img, img_size)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image).astype(np.float32) / 255.0

    return img, image, ratio, dwdh

#  Inferencia genérica para cualquier modelo ONNX
def infer(img_path, model_path, class_names_path=None, use_cuda=True):
    session, class_names, colors = load_model(model_path, class_names_path, use_cuda)
    ori_img, image, ratio, dwdh = preprocess(img_path)
    
    inname = [i.name for i in session.get_inputs()]
    outname = [i.name for i in session.get_outputs()]
    inp = {inname[0]: image}

    outputs = session.run(outname, inp)[0]

    for batch_id, x0, y0, x1, y1, cls_id, score in outputs:
        box = np.array([x0, y0, x1, y1]) - np.array(dwdh * 2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 3)
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        color = colors[name]
        label = f"{name} {score}"
        cv2.rectangle(ori_img, box[:2], box[2:], color, 2)
        cv2.putText(ori_img, label, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (225, 255, 255), 2)

    return Image.fromarray(ori_img)

#  Probar con cualquier modelo y clases
img_result = infer(
    img_path="models/horses.jpg",
    model_path="models/yolov7-tiny.onnx",
    class_names_path="models/clases.json",
    use_cuda=True
)
img_result.show()