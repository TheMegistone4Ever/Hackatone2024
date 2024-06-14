import os

import cv2


def crop_images(root_folder):
    annotations_folder = os.path.join(root_folder, "annotations")
    os.makedirs(annotations_folder, exist_ok=True)

    k = 1
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            images_folder = os.path.join(folder_path, "images")
            labels_folder = os.path.join(folder_path, "labels")
            if os.path.exists(images_folder) and os.path.exists(labels_folder):
                for image_name in os.listdir(images_folder):
                    image_path = os.path.join(images_folder, image_name)
                    label_path = os.path.join(labels_folder, image_name.replace(".jpg", ".txt"))
                    if os.path.isfile(image_path) and os.path.isfile(label_path):
                        try:
                            image = cv2.imread(image_path)
                            image_h, image_w, _ = image.shape
                            with (open(label_path, "r") as f):
                                lines = f.readlines()
                                for line in lines:
                                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                                    x_center, y_center, width, height = int(x_center * image_w), int(
                                        y_center * image_h), int(width * image_w), int(height * image_h)
                                    x_min, y_min = x_center - width // 2, y_center - height // 2
                                    x_max, y_max = x_center + width // 2, y_center + height // 2
                                    area = width * height
                                    if area <= 0 or x_min < 0 or y_min < 0 or x_max > image_w or y_max > image_h:
                                        continue
                                    cropped_image = image[y_min:y_max, x_min:x_max]
                                    cropped_image_name = f"{k}_{folder_name}_{image_name.split(".")[0]}_{int(class_id)}.jpg"
                                    cv2.imwrite(os.path.join(annotations_folder, cropped_image_name), cropped_image)
                                    print(f"{k}. Image {cropped_image_name} saved!")
                                    k += 1
                        except Exception as e:
                            print(f"Error: {e}")
                        finally:
                            continue


if __name__ == "__main__":
    root_folder_input = input("Enter the root folder path: ")
    crop_images(root_folder_input)
