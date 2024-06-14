from numpy import asarray
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("../models/best_10.pt")

    results = model(["../images/im3.jpg", "../images/im3_after.jpg"])

    for result in results:
        boxes = result.boxes
        print(boxes)
        print(int(boxes.xywh[0][1]))
        masks = result.masks
        key_points = result.keypoints
        probs = result.probs
        obb = result.obb
        result.show()
        print([asarray(boxes.xywh).astype(int) for boxes in result.boxes])


def get_boxes_classes(yolo_model, image_path):
    result = yolo_model(image_path)[0].boxes
    classes = result.cls
    return [asarray(boxes.xywh).astype(int) for boxes in result], asarray(classes).astype(int)
