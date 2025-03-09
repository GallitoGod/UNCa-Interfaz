import torch
import os

# Cargar modelo de YOLOv7 en PyTorch
model_path = os.path.abspath("models/best_4class.pt")
model = torch.load(model_path, map_location="cpu")
model.eval()

# Crear un tensor de entrada ficticio
dummy_input = torch.randn(1, 3, 640, 640)

# Intentar exportar a ONNX
try:
    torch.onnx.export(
        model, dummy_input, "yolov7-custom.onnx",
        opset_version=11, input_names=["input"], output_names=["output"]
    )
    print("El modelo se exportó correctamente a ONNX.")
except Exception as e:
    print("Falló la exportación a ONNX:", e)
