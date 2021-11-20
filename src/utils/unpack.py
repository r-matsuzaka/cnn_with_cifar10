import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

home_path = Path(__file__).parents[2]
train_source_path = home_path / "cifar-10-batches-py/data_batch_"
test_source_path = home_path / "cifar-10-batches-py/test_batch"

data_path = home_path / "data"
train_data_path = home_path / "data/train"
test_data_path = home_path / "data/test"
img_size = 1024


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


def image_save(source_path: str, mode: str, batch_id: int = 0) -> None:
    """
    Load images and save labels

    Args:
        source_path (str): path to cifar10 data
        mode (str): train or test
        batch_id (int): batch id

    """
    print(f"mode:{mode}")
    save_data_path = data_path / f"{mode}"

    label_path = data_path / f"{mode}_label.txt"

    # one batch size
    num_images = 10000
    if mode == "train":
        source_path = source_path + str(batch_id + 1)

    # Set initial no
    image_id = batch_id * num_images
    print(f"batch_id:{batch_id}")
    print(f"initial id:{image_id}")
    print("*******************")

    with open(source_path, "rb") as fd, open(label_path, mode="a") as fl:
        dict = pickle.load(fd, encoding="bytes")

        rgb_lst = dict[b"data"]
        label_lst = dict[b"labels"]

        for i in range(num_images):
            rgb_arr = create_rgb_array(rgb_lst[i])
            pil_image_color = Image.fromarray(rgb_arr)

            pil_image_color.save(save_data_path / f"{mode}_{image_id}.png")
            print(f"{mode}_{image_id}.png")
            fl.write(f"{label_lst[i]}")
            image_id += 1


image_save(str(test_source_path), mode="test")

n_batch = 5
for i in range(n_batch):
    image_save(str(train_source_path), mode="train", batch_id=i)
