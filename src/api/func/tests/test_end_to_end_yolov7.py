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
    img_path = Path("api/func/tests/testing_images/horses.jpg")
    image = cv2.imread(str(img_path))
    assert image is not None, "No se pudo cargar la imagen de prueba."

    model_path = Path("models/yolov7-tiny.json")
    controller = ModelController()
    controller.load_model(model_path)
    result = controller.process(image)

    boxes = result["boxes"]
    labels = result.get("labels", [])
    scores = result.get("scores", [])

    output_img = draw_boxes(image.copy(), boxes, labels, scores)

    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Detecciones YOLOv7-tiny")
    plt.axis("off")
    plt.show()

    cv2.imwrite("api/func/tests/testing_images/horses_detected.png", output_img)
