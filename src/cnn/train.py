import glob
import os
from pathlib import Path

import numpy as np
from natsort import natsorted
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

print(os.path.basename(__file__))
import sys

sys.path.append("/home/ryo/cnn_with_cifar10/src/utils")


import helpers
import visualize

home_path = Path(__file__).parents[2]
train_data_path = home_path / "data/train"
test_data_path = home_path / "data/test"
data_path = home_path / "data"
train_label_path = data_path / "train_label.txt"
test_label_path = data_path / "test_label.txt"

train_size = 50000
test_size = 10000


def yes_no_input():
    while True:
        choice = input(
            "This calculation takes much time. If you want to continue calculation respond with 'yes', otherwise'no' [y/N]: "
        ).lower()
        if choice in ["y", "ye", "yes"]:
            return True
        elif choice in ["n", "no"]:
            return False


@helpers.timer
def load_image(num_image: int, image_paths: list) -> np.ndarray:
    """
    load image

    Args:
        num_image (int): number of images
        image_paths (list): path list to a image


    Returns:
        images  (np.ndarray): numpy array of rgb value.
        shape for train is (50000,32,32).
        shape for test is (10000,32,32).
    """

    images = np.zeros((num_image, 32, 32, 3), "uint8")

    for i, image in enumerate(image_paths):
        img = Image.open(image)
        img_array = np.asarray(img)

        images[i, ...] = img_array

    print(images.shape)

    return images


@helpers.timer
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

    print(labels_arr.shape)

    return labels_arr


@helpers.timer
def LeNet(models):
    """ "
    Lenet architecture

    """
    num_classes = 10
    print("For cnn architecture LeNet is used.")
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            filters=6,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            activation="tanh",
            input_shape=(32, 32, 3),
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation="tanh"))
    model.add(layers.Dense(84, activation="tanh"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.summary()

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


@helpers.timer
def load_data():
    """ "
    Load data
    """

    print("Loading paths...")

    train_file_paths = natsorted(glob.glob(str(train_data_path / "*")))
    test_file_paths = natsorted(glob.glob(str(test_data_path / "*")))

    print("Train image size:")
    train_images = load_image(train_size, train_file_paths)
    print("Test image size:")
    test_images = load_image(test_size, test_file_paths)

    print("Train label size:")
    train_labels = load_label(str(train_label_path))
    print("Test label size:")
    test_labels = load_label(str(test_label_path))

    train_labels_onehot = to_categorical(train_labels, 10)
    test_labels_onehot = to_categorical(test_labels, 10)

    return train_images, test_images, train_labels_onehot, test_labels_onehot


@helpers.timer
def normalization(train_images, test_images):
    """
    Noarmalization
    """
    print("Normalization...")
    train_images = train_images.astype("float32")
    test_images = test_images.astype("float32")

    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images, test_images


@helpers.timer
def training(model):
    """
    Training
    """

    print("Training model...")

    history = model.fit(
        train_images,
        train_labels_onehot,
        batch_size=32,
        epochs=10,
        validation_split=0.1,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(test_images, test_labels_onehot)

    return test_acc, history


if yes_no_input():

    print(
        "OK. The training cnn with cifiar10 dataset for classification is started with CPU."
    )

    # データのロード
    train_images, test_images, train_labels_onehot, test_labels_onehot = load_data()

    # ピクセルの値を 0~1 の間に正規化
    train_images, test_images = normalization(train_images, test_images)

    # LeNet
    model = LeNet(models)

    # Training
    test_acc, history = training(model)

    print("Visualization...")
    visualize.save_fig(history)

    print(test_acc)
