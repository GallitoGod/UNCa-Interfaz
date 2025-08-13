import cv2
import numpy as np
from pathlib import Path
from api.func.model_controller import ModelController
import matplotlib.pyplot as plt

def draw_boxes(image, boxes, labels=None, scores=None):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = labels[i] if labels else ""
        score = scores[i] if scores else ""
        text = f"{label} {score:.2f}" if score != "" else str(label)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def test_yolov7_tiny_on_horse_image():
    img_path = Path(__file__).parent / "testing_images" / "imagen_redimensionada.jpg"
    image = cv2.imread(str(img_path))
    assert image is not None, "No se pudo cargar la imagen de prueba."

    model_path = Path("models/yolov7-tiny.onnx")
    controller = ModelController()
    controller.load_model(model_path)
    result = controller.inference(image)
    h, w = image.shape[:2]
    print("Total detecciones (len result):", len(result))
    for i, det in enumerate(result):
        det = list(det)
        print(f"[{i}] raw:", det)
        # detectar si el primer valor es batch_id (7 elementos esperados: batch,x1,y1,x2,y2,cls,score)
        if len(det) == 7:
            _, x1,y1,x2,y2, cls, score = det
        elif len(det) >= 6:
            x1,y1,x2,y2, cls, score = det[:6]
        else:
            print("Formato inesperado:", len(det))
            continue

        x1i,y1i,x2i,y2i = map(int, (round(x1), round(y1), round(x2), round(y2)))
        x1c, y1c = max(0, x1i), max(0, y1i)
        x2c, y2c = min(w-1, x2i), min(h-1, y2i)
        area = max(0, x2c - x1c) * max(0, y2c - y1c)
        print(f"  clase={int(cls)}, score={score:.3f}, box_raw=({x1i},{y1i},{x2i},{y2i}), clipped=({x1c},{y1c},{x2c},{y2c}), area={area}")

    boxes = []
    labels = []
    scores = []

    for item in result:
        boxes.append(item[0:4])
        labels.append(item[4])
        scores.append(item[5])

    output_img = draw_boxes(image.copy(), boxes, labels, scores)

    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Detecciones YOLOv7-tiny")
    plt.axis("off")
    plt.show()

    cv2.imwrite(Path(__file__).parent / "testing_images" / "horses_detected.jpg", output_img)
