import glob
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import datasets, layers, models

home_path = Path(__file__).parents[2]
train_data_path = home_path / "data/train"
test_data_path = home_path / "data/test"
data_path = home_path / "data"
train_label_path = data_path / "train_label.txt"
test_label_path = data_path / "test_label.txt"

train_size = 8000
test_size = 2000


def load_image(num_image: int, image_paths: list) -> np.ndarray:
    """
    load image

    Args:
        num_image (int): number of images
        image_paths (list): path list to a image


    Returns:
        images  (np.ndarray): numpy array of rgb value.
        shape for train is (8000,32,32).
        shape for test is (2000,32,32).
    """

    images = np.zeros((num_image, 32, 32, 3), "uint8")

    for i, image in enumerate(image_paths):
        img = Image.open(image)
        img_array = np.asarray(img)

        images[i, ...] = img_array

    print(images.shape)

    return images


def load_label(label_path: str) -> np.ndarray:
    """
    load label

    Args:
        rgb_lst (list): list of rgb value


    Returns:
        labels_arr  (np.ndarray): numpy array of labels.
        shape for train is (8000,).
        shape for test is (2000,).
    """

    with open(label_path) as f:
        labels = f.read()
        label_lst = list(labels)
        label_lst = [int(i) for i in label_lst]
        labels_arr = np.array(label_lst)

    return labels_arr


train_file_paths = glob.glob(str(train_data_path / "*"))
test_file_paths = glob.glob(str(test_data_path / "*"))

train_images = load_image(train_size, train_file_paths)
test_images = load_image(test_size, test_file_paths)

train_labels = load_label(str(train_label_path))
test_labels = load_label(str(test_label_path))
