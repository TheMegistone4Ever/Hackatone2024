from __future__ import absolute_import, division, print_function

from time import time

import joblib
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO

from utils.depth import load_model, process_and_plot
from utils.kmeans_image_clustering import clusterize, get_score
from utils.yolo_inference import get_boxes_classes

if __name__ == "__main__":
    encoder, decoder = load_model()  # Завантажуємо модель mono_stereo_640x192
    depth = process_and_plot(["../images/im3.jpg",
                              "../images/im3_after.jpg"])  # тут path до фото з ушкодженою будівлею(глибина ушкодженої будівлі)
    start = time()
    model = YOLO("../models/best.pt")  # завантажуємо модель YOLO як best.pt
    bboxes, classes = get_boxes_classes(model, "../images/im3_after.jpg")  # path до зображення ушкодженої будівлі
    print(f"Bounding boxes{bboxes}")

    model_path = "../models/kmeans_model_k=200_random_state=1810_86x86.pkl"

    print("Loading cluster KMeans model...")
    kmeans = joblib.load(model_path)

    cluster = clusterize("../images/im3_after.jpg", kmeans)
    print(f"{cluster=}")
    for bbox, yolo_class in zip(bboxes, classes):
        x_center, y_center, width, height = bbox[0]
        x_min, y_min = x_center - width // 2, y_center - height // 2
        x_max, y_max = x_center + width // 2, y_center + height // 2

        aoi = depth[int(y_min):int(y_max), int(x_min):int(x_max)]

        sum_aoi = aoi.sum()
        diff_min_max = aoi.max() - aoi.min()
        score = sum_aoi / (aoi.shape[0] * aoi.shape[1] * diff_min_max)
        print(f"{score=}")

        roi_depth = depth[int(y_min):int(y_max), int(x_min):int(x_max)]
        average_depth = np.mean(roi_depth)

        # Переведення розмірів пікселів в метри
        pixel_width = x_max - x_min
        pixel_height = y_max - y_min

        # Припустимо, що кожен піксель відповідає real_world_meters на кожен метр глибини
        real_world_meters_per_pixel = 0.00125  # Це значення має бути каліброване або визначене експериментально

        object_width_meters = pixel_width * real_world_meters_per_pixel * average_depth
        object_height_meters = pixel_height * real_world_meters_per_pixel * average_depth

        print("Приблизна ширина об'єкта в метрах:", object_width_meters)
        print("Приблизна висота об'єкта в метрах:", object_height_meters)

        area = object_width_meters * object_height_meters
        print(f"Приблизна площа об'єкта: {area} м^2")

        print(f"Зважена оцінка: {get_score(yolo_class, cluster, score)}")

        print(f"Time: {time() - start}")

    plt.show()
