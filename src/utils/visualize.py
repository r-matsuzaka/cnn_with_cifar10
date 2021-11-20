import os
from pathlib import Path

import matplotlib.pyplot as plt

home_path = Path(__file__).parents[2]
result_path = home_path / "results"


def create_dir() -> None:
    """
    create direcotry for results
    """
    os.makedirs(result_path, exist_ok=True)


def plot_history_loss(fit, ax):
    # Plot the loss in the history
    ax.plot(fit.history["loss"], label="loss for training")
    ax.plot(fit.history["val_loss"], label="loss for validation")
    ax.set_title("model loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")


def plot_history_acc(fit, ax):
    # Plot the loss in the history
    ax.plot(fit.history["accuracy"], label="accuracy for training")
    ax.plot(fit.history["val_accuracy"], label="accuracy for validation")
    ax.set_title("model accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend(loc="upper right")


def save_fig(history):
    create_dir()
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))

    axL = axs[0]
    axR = axs[1]

    plot_history_loss(history, axL)
    plot_history_acc(history, axR)
    fig.savefig(result_path / "loss_gragh.png")

    plt.close()
