import time
from pathlib import Path
from PIL import Image
import numpy as np

from api.func.model_controller import ModelController

MODEL_PATH = "models/yolov7-tiny.onnx"
IMAGE_PATH = Path(__file__).parent / "testing_images" / "imagen_redimensionada.jpg"
NUM_IMAGES = 100  

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def benchmark(model_path, image_path, num_images):
    t0 = time.perf_counter()
    controller = ModelController()
    controller.load_model(model_path)
    t1 = time.perf_counter()
    load_time = t1 - t0
    print(f"Tiempo de carga total: {load_time:.3f} s")

    image = load_image(image_path)

    controller.inference(image)

    t0 = time.perf_counter()
    controller.inference(image)
    t1 = time.perf_counter()
    infer_time = t1 - t0
    print(f"Tiempo por imagen: {infer_time:.3f} s")

    t0 = time.perf_counter()
    for _ in range(num_images):
        controller.inference(image)
    t1 = time.perf_counter()
    total_time = t1 - t0
    fps = num_images / total_time
    print(f"FPS estimados: {fps:.2f}")

if __name__ == "__main__":
    benchmark(MODEL_PATH, IMAGE_PATH, NUM_IMAGES)
