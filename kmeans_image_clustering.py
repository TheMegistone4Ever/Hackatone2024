import os
import shutil

import joblib
import numpy as np
from PIL import Image
from numpy import dot
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def lsh_clustering(folder_path, k, from_pretrained=False, image_size=(86, 86), random_state=1810):
    print("Creating cluster folders...")
    for i in range(k):
        cluster_folder = os.path.join(folder_path, f"Cluster_{i + 1}")
        os.makedirs(cluster_folder, exist_ok=True)

    print("Reading images...")
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    images = []
    for i, file in enumerate(image_files):
        image_path = os.path.join(folder_path, file)
        image = Image.open(image_path)
        images.append(np.array(image.resize(image_size)).flatten())
        print(f"{i + 1}. Image {file} read!")

    images = np.array(images)

    folder_model_path = os.path.join(folder_path, "KMeans_Models")
    model_path = os.path.join(folder_model_path,
                              f"kmeans_model_{k=}_{random_state=}_{image_size[0]}x{image_size[1]}.pkl")

    if from_pretrained:
        print("Loading cluster KMeans model...")
        kmeans = joblib.load(model_path)
    else:
        print("Clustering images...")
        kmeans = KMeans(n_clusters=k, random_state=random_state).fit(images)
        print("Saving cluster KMeans model...")
        os.makedirs(folder_model_path, exist_ok=True)
        joblib.dump(kmeans, model_path)
        print(f"Model saved at {model_path}...")

    print("Copying images to respective clusters...")
    closest, _ = pairwise_distances_argmin_min(images, kmeans.cluster_centers_)

    for i, idx in enumerate(closest):
        image_file = image_files[i]
        src_path = os.path.join(folder_path, image_file)
        dest_folder = os.path.join(folder_path, f"Cluster_{kmeans.labels_[idx] + 1}")
        shutil.copy(src_path, dest_folder)
        print(f"{i + 1}. Image {image_file} copied to {dest_folder}")


def clusterize(image, kmeans, size=(86, 86)):
    if isinstance(image, str):
        image = np.array(Image.open(image).resize(size)).flatten()
    elif isinstance(image, Image.Image):
        image = np.array(image.resize(size)).flatten()
    prediction = kmeans.predict([image])
    return prediction[0]


def map_yolo_class_to_damage(yolo_class, nc=5):
    return 1 - yolo_class / nc


def map_cluster_to_damage(cluster, k=200):
    if k != 200:
        return .0
    if cluster == 0:
        return .3
    elif cluster == 1:
        return .6
    elif cluster == 2:
        return .2
    elif cluster == 3:
        return .001
    elif cluster == 4:
        return .001
    elif cluster == 5:
        return .001
    elif cluster == 6:
        return .7
    elif cluster == 7:
        return .001
    elif cluster == 8:
        return .001
    elif cluster == 9:
        return .2
    elif cluster == 10:
        return .1
    elif cluster == 11:
        return .001
    elif cluster == 12:
        return .001
    elif cluster == 13:
        return .3
    elif cluster == 14:
        return .001
    elif cluster == 15:
        return .001
    elif cluster == 16:
        return .0015
    elif cluster == 17:
        return .001
    elif cluster == 18:
        return .001
    elif cluster == 19:
        return .001
    elif cluster == 20:
        return .001
    elif cluster == 21:
        return .2
    elif cluster == 22:
        return .001
    elif cluster == 23:
        return .001
    elif cluster == 24:
        return .001
    elif cluster == 25:
        return .001
    elif cluster == 26:
        return .001
    elif cluster == 27:
        return .001
    elif cluster == 28:
        return .0015
    elif cluster == 29:
        return .001
    elif cluster == 30:
        return .001
    elif cluster == 31:
        return .001
    elif cluster == 32:
        return .001
    elif cluster == 33:
        return .1
    elif cluster == 34:
        return .001
    elif cluster == 35:
        return .001
    elif cluster == 36:
        return .001
    elif cluster == 37:
        return .001
    elif cluster == 38:
        return .15
    elif cluster == 39:
        return .9
    elif cluster == 40:
        return .5
    elif cluster == 41:
        return .3
    elif cluster == 42:
        return .001
    elif cluster == 43:
        return .001
    elif cluster == 44:
        return .9
    elif cluster == 45:
        return .25
    elif cluster == 46:
        return .001
    elif cluster == 47:
        return .001
    elif cluster == 48:
        return .8
    elif cluster == 49:
        return .001
    elif cluster == 50:
        return .001
    elif cluster == 51:
        return .6
    elif cluster == 52:
        return .001
    elif cluster == 53:
        return .001
    elif cluster == 54:
        return .001
    elif cluster == 55:
        return .5
    elif cluster == 56:
        return .001
    elif cluster == 57:
        return .001
    elif cluster == 58:
        return .001
    elif cluster == 59:
        return .001
    elif cluster == 60:
        return .001
    elif cluster == 61:
        return .001
    elif cluster == 62:
        return .001
    elif cluster == 63:
        return .001
    elif cluster == 64:
        return .001
    elif cluster == 65:
        return .001
    elif cluster == 66:
        return .001
    elif cluster == 67:
        return .001
    elif cluster == 68:
        return .001
    elif cluster == 69:
        return .001
    elif cluster == 70:
        return .001
    elif cluster == 71:
        return .1
    elif cluster == 72:
        return .001
    elif cluster == 73:
        return .001
    elif cluster == 74:
        return .001
    elif cluster == 75:
        return .001
    elif cluster == 76:
        return .3
    elif cluster == 77:
        return .001
    elif cluster == 78:
        return .8
    elif cluster == 79:
        return .001
    elif cluster == 80:
        return .001
    elif cluster == 81:
        return .001
    elif cluster == 82:
        return .001
    elif cluster == 83:
        return .001
    elif cluster == 84:
        return .001
    elif cluster == 85:
        return .2
    elif cluster == 86:
        return .001
    elif cluster == 87:
        return .001
    elif cluster == 88:
        return .4
    elif cluster == 89:
        return .001
    elif cluster == 90:
        return .001
    elif cluster == 91:
        return .001
    elif cluster == 92:
        return .75
    elif cluster == 93:
        return .001
    elif cluster == 94:
        return .001
    elif cluster == 95:
        return .5
    elif cluster == 96:
        return .001
    elif cluster == 97:
        return .001
    elif cluster == 98:
        return .15
    elif cluster == 99:
        return .1
    elif cluster == 100:
        return .001
    elif cluster == 101:
        return .001
    elif cluster == 102:
        return .7
    elif cluster == 103:
        return .001
    elif cluster == 104:
        return .001
    elif cluster == 105:
        return .001
    elif cluster == 106:
        return .001
    elif cluster == 107:
        return .001
    elif cluster == 108:
        return .001
    elif cluster == 109:
        return .2
    elif cluster == 110:
        return .001
    elif cluster == 111:
        return .001
    elif cluster == 112:
        return .3
    elif cluster == 113:
        return .001
    elif cluster == 114:
        return .001
    elif cluster == 115:
        return .1
    elif cluster == 116:
        return .001
    elif cluster == 117:
        return .001
    elif cluster == 118:
        return .001
    elif cluster == 119:
        return .001
    elif cluster == 120:
        return .001
    elif cluster == 121:
        return .001
    elif cluster == 122:
        return .001
    elif cluster == 123:
        return .65
    elif cluster == 124:
        return .001
    elif cluster == 125:
        return .001
    elif cluster == 126:
        return .3
    elif cluster == 127:
        return .001
    elif cluster == 128:
        return .6
    elif cluster == 129:
        return .001
    elif cluster == 130:
        return .001
    elif cluster == 131:
        return .001
    elif cluster == 132:
        return .2
    elif cluster == 133:
        return .001
    elif cluster == 134:
        return .1
    elif cluster == 135:
        return .001
    elif cluster == 136:
        return .001
    elif cluster == 137:
        return .001
    elif cluster == 138:
        return .001
    elif cluster == 139:
        return .001
    elif cluster == 140:
        return .1
    elif cluster == 141:
        return .001
    elif cluster == 142:
        return .001
    elif cluster == 143:
        return .001
    elif cluster == 144:
        return .001
    elif cluster == 145:
        return .001
    elif cluster == 146:
        return .001
    elif cluster == 147:
        return .001
    elif cluster == 148:
        return .001
    elif cluster == 149:
        return .001
    elif cluster == 150:
        return .4
    elif cluster == 151:
        return .001
    elif cluster == 152:
        return .001
    elif cluster == 153:
        return .001
    elif cluster == 154:
        return .001
    elif cluster == 155:
        return .65
    elif cluster == 156:
        return .3
    elif cluster == 157:
        return .001
    elif cluster == 158:
        return .8
    elif cluster == 159:
        return .2
    elif cluster == 160:
        return .9
    elif cluster == 161:
        return .001
    elif cluster == 162:
        return .001
    elif cluster == 163:
        return .001
    elif cluster == 164:
        return .001
    elif cluster == 165:
        return .001
    elif cluster == 166:
        return .001
    elif cluster == 167:
        return .001
    elif cluster == 168:
        return .001
    elif cluster == 169:
        return .001
    elif cluster == 170:
        return .001
    elif cluster == 171:
        return .7
    elif cluster == 172:
        return .001
    elif cluster == 173:
        return .001
    elif cluster == 174:
        return .001
    elif cluster == 175:
        return .001
    elif cluster == 176:
        return .4
    elif cluster == 177:
        return .001
    elif cluster == 178:
        return .001
    elif cluster == 179:
        return .5
    elif cluster == 180:
        return .001
    elif cluster == 181:
        return .001
    elif cluster == 182:
        return .001
    elif cluster == 183:
        return .001
    elif cluster == 184:
        return .35
    elif cluster == 185:
        return .001
    elif cluster == 186:
        return .8
    elif cluster == 187:
        return .001
    elif cluster == 188:
        return .001
    elif cluster == 189:
        return .1
    elif cluster == 190:
        return .001
    elif cluster == 191:
        return .1
    elif cluster == 192:
        return .001
    elif cluster == 193:
        return .2
    elif cluster == 194:
        return .001
    elif cluster == 195:
        return .001
    elif cluster == 196:
        return .6
    elif cluster == 197:
        return .001
    elif cluster == 198:
        return .001
    elif cluster == 199:
        return .9


def get_score(yolo_class, cluster, norm_diff, weights=(.2, .1, .7)):
    damage = map_yolo_class_to_damage(yolo_class)
    damage_cluster = map_cluster_to_damage(cluster)
    return dot([damage, damage_cluster, norm_diff], weights)


if __name__ == "__main__":
    print("Welcome to the Image Clustering using KMeans!")
    folder_path_input = input("Enter the folder path: ")
    k_input = int(input("Enter the number of clusters (k): "))
    lsh_clustering(folder_path_input, k_input, True)
