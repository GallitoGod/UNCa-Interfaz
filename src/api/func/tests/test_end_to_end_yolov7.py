import cv2
import numpy as np
from pathlib import Path
from api.func.model_controller import ModelController
import matplotlib.pyplot as plt


def draw_boxes(image, boxes, labels=None, scores=None):
    """
    boxes: iterable de [x1,y1,x2,y2] en pixeles.
    labels: iterable de class_id (int)
    scores: iterable de confidence (float)
    """
    h, w = image.shape[:2]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = map(int, map(round, (x1, y1, x2, y2)))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # clamp
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if (x2 - x1) <= 1 or (y2 - y1) <= 1:
            continue

        label = int(labels[i]) if labels is not None else -1
        score = float(scores[i]) if scores is not None else None

        if score is not None:
            text = f"{label} {score:.2f}"
        else:
            text = str(label)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            text,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return image


def test_yolov7_tiny_on_horse_image():
    img_path = Path(__file__).parent / "testing_images" / "imagen_redimensionada.jpg"
    image = cv2.imread(str(img_path))
    assert image is not None, "No se pudo cargar la imagen de prueba."

    h, w = image.shape[:2]
    print(f"Imagen: {w}x{h}")

    model_path = Path("models/yolov7-tiny.onnx")
    controller = ModelController()
    controller.load_model(model_path)

    result = controller.inference(image)
    
    print("Total detecciones (len result):", len(result))

    boxes, labels, scores = [], [], []
    for i, det in enumerate(result):
        det = list(det)

        if len(det) < 6:
            print(f"[{i}] Formato inesperado (len={len(det)}): {det}")
            continue

        x1, y1, x2, y2, conf, cls = det[:6]

        x1i, y1i, x2i, y2i = map(int, map(round, (x1, y1, x2, y2)))

        if x2i < x1i:
            x1i, x2i = x2i, x1i
        if y2i < y1i:
            y1i, y2i = y2i, y1i

        x1c, y1c = max(0, x1i), max(0, y1i)
        x2c, y2c = min(w - 1, x2i), min(h - 1, y2i)
        area = max(0, x2c - x1c) * max(0, y2c - y1c)

        print(
            f"[{i}] cls={int(cls)} conf={float(conf):.3f} "
            f"box_raw=({x1i},{y1i},{x2i},{y2i}) clipped=({x1c},{y1c},{x2c},{y2c}) area={area}"
        )

        boxes.append([x1, y1, x2, y2])
        labels.append(int(cls))
        scores.append(float(conf))

    output_img = draw_boxes(image.copy(), boxes, labels, scores)

    plt.figure()
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Detecciones YOLOv7-tiny")
    plt.axis("off")
    plt.show()

    out_path = Path(__file__).parent / "testing_images" / "horses_detected.jpg"
    cv2.imwrite(str(out_path), output_img)
    print("Guardado:", out_path)


if __name__ == "__main__":
    test_yolov7_tiny_on_horse_image()