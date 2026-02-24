import time
from pathlib import Path
import cv2

from api.func.model_controller import ModelController

MODEL_PATH = "models/efficientdet-lite0.tflite"   # "models/efficientdet-lite0.tflite" |  "models/yolov7-tiny.onnx"
IMAGE_PATH = Path(__file__).parent / "testing_images" / "imagen_redimensionada.jpg"

WARMUP_RUNS = 10
NUM_IMAGES = 200

def load_image_cv2(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return img

def benchmark(model_path: str, image_path: Path, warmup_runs: int, num_images: int):
    # --- Load ---
    t0 = time.perf_counter()
    controller = ModelController()

    controller.load_model(model_path)

    t1 = time.perf_counter()
    print(f"Tiempo de carga total: {(t1 - t0)*1000:.2f} ms")

    # --- Image ---
    image = load_image_cv2(image_path)

    # --- Make logs more frequent (opcional) ---
    if hasattr(controller, "_log_every"):
        controller._log_every = 25  
    if hasattr(controller, "_frame_idx"):
        controller._frame_idx = 0

    # --- Warmup ---
    for _ in range(max(0, warmup_runs)):
        controller.inference(image)

    # --- Single inference timing (wall) ---
    t0 = time.perf_counter()
    _ = controller.inference(image)
    t1 = time.perf_counter()
    print(f"Tiempo (wall) por 1 imagen: {(t1 - t0)*1000:.2f} ms")

    # --- Batch FPS (wall) ---
    t0 = time.perf_counter()
    for _ in range(num_images):
        _ = controller.inference(image)
    t1 = time.perf_counter()

    total_s = (t1 - t0)
    fps = (num_images / total_s) if total_s > 0 else float("inf")
    print(f"Total {num_images} imgs: {total_s:.3f} s | FPS (wall): {fps:.2f}")

    # --- Optional: print controller perf stats if exposed ---
    if hasattr(controller, "perf") and hasattr(controller.perf, "stats"):
        s = controller.perf.stats()
        if s:
            print(
                "PerfMeter | n={n} avg={avg_ms:.2f}ms p95={p95_ms:.2f}ms fps={fps_avg:.2f} | "
                "pre={pre_avg_ms:.2f} inf={inf_avg_ms:.2f} post={post_avg_ms:.2f}".format(**s)
            )

if __name__ == "__main__":
    benchmark(MODEL_PATH, IMAGE_PATH, WARMUP_RUNS, NUM_IMAGES)