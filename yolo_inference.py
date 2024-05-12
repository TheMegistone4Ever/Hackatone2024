from ultralytics import YOLO
from numpy import asarray

model = YOLO("models/best_10.pt")

results = model(["im3.jpg", "im3_after.jpg"])

for result in results:
    boxes = result.boxes
    print(boxes)
    # print(int(boxes.xywh[0][1]))
    # masks = result.masks
    # key_points = result.keypoints
    # probs = result.probs
    # obb = result.obb
    # result.show()
    print([asarray(boxes.xywh).astype(int) for boxes in result.boxes])
