import torch
import time
from pathlib import Path
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torchUtils import select_device, TracedModel

def load_model(weights_path, device='cpu', img_size=640):
    """
    Carga el modelo YOLOv7 en el dispositivo especificado.
    """
    device = select_device(device)
    model = attempt_load(weights_path, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    model = TracedModel(model, device, img_size)
    model.half() if device.type != 'cpu' else model.float()
    return model, device, img_size

def predict(model, device, img, conf_thres=0.25, iou_thres=0.45):
    """
    Realiza predicciones en una imagen dada usando el modelo YOLOv7.
    """
    img = torch.from_numpy(img).to(device)
    img = img.half() if device.type != 'cpu' else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    with torch.no_grad():
        pred = model(img)[0]
    
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    detections = []
    
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (img.shape[2], img.shape[3])).round()
            for *xyxy, conf, cls in det:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / torch.tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2]])).view(-1).tolist()
                detections.append({'class': int(cls), 'xywh': xywh, 'confidence': float(conf)})
    
    return detections
