import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

home_path = Path(__file__).parents[2]
source_path = home_path / "cifar-10-batches-py/data_batch_1"
data_path = home_path / "data"
train_data_path = home_path / "data/train"
test_data_path = home_path / "data/test"
img_size = 1024
train_size = 8000


def create_dir() -> None:
    """
    create direcotry for train and test
    """

    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(test_data_path, exist_ok=True)


def create_rgb_array(rgb_lst: list) -> np.ndarray:
    """
    Create numpy array of rgb value

    Args:
        rgb_lst (list): list of rgb value


    Returns:
        rgb_arr (np.ndarray): numpy array of rgb value. shape is (32,32,3).

    """

    create_dir()

    r_lst = rgb_lst[:img_size]
    r_arr = np.array(r_lst)
    r_arr = r_arr.reshape([32, 32])

    g_lst = rgb_lst[img_size : 2 * img_size]
    g_arr = np.array(g_lst)
    g_arr = g_arr.reshape([32, 32])

    b_lst = rgb_lst[2 * img_size :]
    b_arr = np.array(b_lst)
    b_arr = b_arr.reshape([32, 32])

    rgb_arr = np.zeros((32, 32, 3), "uint8")
    rgb_arr[..., 0] = r_arr
    rgb_arr[..., 1] = g_arr
    rgb_arr[..., 2] = b_arr

    return rgb_arr


train_label_path = data_path / "train_label.txt"
test_label_path = data_path / "test_label.txt"

with open(source_path, "rb") as fd, open(train_label_path, mode="w") as ftr, open(
    test_label_path, mode="w"
) as fte:
    dict = pickle.load(fd, encoding="bytes")

    rgb_lst = dict[b"data"]
    label_lst = dict[b"labels"]

    for i in range(train_size):
        rgb_arr = create_rgb_array(rgb_lst[i])
        pil_image_color = Image.fromarray(rgb_arr)

        pil_image_color.save(train_data_path / f"train_{i}.png")
        ftr.write(f"{label_lst[i]}")

    j = 0
    for i in range(train_size, 10000):
        rgb_arr = create_rgb_array(rgb_lst[i])
        pil_image_color = Image.fromarray(rgb_arr)

        pil_image_color.save(test_data_path / f"test_{j}.png")
        fte.write(f"{label_lst[i]}")
        j += 1
