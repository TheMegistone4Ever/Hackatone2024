from __future__ import absolute_import, division, print_function

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as pil
from torchvision import transforms

from monodepth2.networks import ResnetEncoder, DepthDecoder
from monodepth2.utils import download_model_if_doesnt_exist

model_name = "mono+stereo_640x192"
download_model_if_doesnt_exist(model_name)
encoder_path = os.path.join("../models", model_name, "encoder.pth")
depth_decoder_path = os.path.join("../models", model_name, "depth.pth")


# Завантаження моделі
def load_model():
    encoder = ResnetEncoder(18, False)
    depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    encoder.eval()
    depth_decoder.eval()
    return encoder, depth_decoder


encoder, depth_decoder = load_model()


def process_and_plot(image_paths, feed_width=640, feed_height=192):
    plt.figure(figsize=(12, 20), dpi=200)
    depths = []
    for idx, image_path in enumerate(image_paths, start=1):
        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

        with torch.no_grad():
            features = encoder(input_image_pytorch)
            outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        depth = 1.0 / (disp + 1e-6)

        depth_resized = torch.nn.functional.interpolate(depth, (original_height, original_width), mode="bilinear",
                                                        align_corners=False)

        depth_resized_np = depth_resized.squeeze().cpu().numpy()
        disparity_resized_np = disp.squeeze().cpu().numpy()

        disparity_resized_np = cv2.resize(disparity_resized_np, (feed_width, 500))
        depth_resized_np = cv2.resize(depth_resized_np, (feed_width, 500))
        depths.append(depth_resized_np)

        plt.subplot(3, 2, idx * 2 - 1)
        plt.imshow(depth_resized_np, cmap='plasma')
        plt.title(f"Depth estimation for {os.path.basename(image_path)}")
        plt.axis('off')

        plt.subplot(3, 2, idx * 2)
        plt.imshow(disparity_resized_np, cmap='plasma')
        plt.title(f"Disparity estimation for {os.path.basename(image_path)}")
        plt.axis('off')

    diff = abs(depths[0] - depths[1])
    diff = cv2.resize(diff, (feed_width * 2, 1000))

    plt.subplot(3, 1, 3)
    plt.imshow(diff, cmap='plasma')
    plt.title("Difference between depths")
    plt.axis('off')

    return diff


if __name__ == "__main__":
    # Обробка та візуалізація для одного зображення
    image_paths = [r"images\im3.jpg", r"images\im3_after.jpg"]
    diff = process_and_plot(image_paths)
    plt.show()

    x_min, y_min = 0, 0
    x_max, y_max = diff.shape[1] / 2, diff.shape[0] / 2

    aoi = diff[int(y_min):int(y_max), int(x_min):int(x_max)]

    sum_aoi = aoi.sum()
    diff_min_max = aoi.max() - aoi.min()
    score = sum_aoi / (aoi.shape[0] * aoi.shape[1] * diff_min_max)
    print(f"{score=}")

    roi_depth = diff[int(y_min):int(y_max), int(x_min):int(x_max)]
    average_depth = np.mean(roi_depth)

    # Переведення розмірів пікселів в метри
    pixel_width = x_max - x_min
    pixel_height = y_max - y_min

    # Припустимо, що кожен піксель відповідає real_world_meters на кожен метр глибини
    real_world_meters_per_pixel = 0.00025  # Це значення має бути каліброване або визначене експериментально

    object_width_meters = pixel_width * real_world_meters_per_pixel * average_depth
    object_height_meters = pixel_height * real_world_meters_per_pixel * average_depth

    print("Приблизна ширина об'єкта в метрах:", object_width_meters)
    print("Приблизна висота об'єкта в метрах:", object_height_meters)

    area = object_width_meters * object_height_meters
