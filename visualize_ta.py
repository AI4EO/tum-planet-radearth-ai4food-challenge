import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from typing import List


def get_ndvi(image_stack):
    if image_stack.shape[1] == 4:
        red = image_stack[:, 2]
        nir = image_stack[:, 3]
    elif image_stack.shape[1] == 13:
        red = image_stack[:, 3]
        nir = image_stack[:, 7]

    ndvi = (nir - red) / (nir + red)
    ndvi[nir == red] = 0
    assert np.isnan(ndvi).sum() == 0, "NDVI contains NaN"
    return ndvi


def get_nir(image_stack):
    if image_stack.shape[1] == 4:
        return image_stack[:, 3]
    elif image_stack.shape[1] == 13:
        return image_stack[:, 7]


def plot_preds(
    title: str,
    model,
    x: torch.Tensor,
    gp_indexes: List[int],
    return_wandb_image: bool = True,
):
    model.gp_inference_indexes = gp_indexes
    seq_length = x.shape[1]

    with torch.no_grad():
        pred = model(x, training=False)
        x_np = x.cpu().numpy()
        pred_np = pred.cpu().numpy()

    input_timesteps = model.input_timesteps

    fig, axes = plt.subplots(nrows=x.shape[0], ncols=2, figsize=(20, x.shape[0] * 7))
    for i in range(x.shape[0]):
        ax = axes[i]
        actual_nir = get_nir(x_np[i])
        actual_ndvi = get_ndvi(x_np[i])
        pred_nir = get_nir(pred_np[i])
        pred_ndvi = get_ndvi(pred_np[i])

        ax[0].plot(actual_nir, label="Actual NIR")
        ax[1].plot(actual_ndvi, label="Actual NDVI")

        ax[0].plot(pred_nir, label="Generated NIR")
        ax[1].plot(pred_ndvi, label="Generated NDVI")

        ax[0].axvline(x=input_timesteps, label="Predictions start", linestyle="--", color="gray")
        ax[1].axvline(x=input_timesteps, label="Predictions start", linestyle="--", color="gray")

        timesteps = [input_timesteps + j for j in gp_indexes if (input_timesteps + j) < seq_length]
        ax[0].plot(timesteps, pred_nir[timesteps], "ro", color="red", label="GP used", markersize=2)
        ax[1].plot(
            timesteps, pred_ndvi[timesteps], "ro", color="red", label="GP used", markersize=2
        )

        ax[0].set_title("NIR")
        ax[0].set_ylabel("NIR")
        ax[0].set_xlabel("Time interval")
        ax[1].set_title("NDVI")
        ax[1].set_ylabel("NDVI")
        ax[1].set_xlabel("Time interval")
        ax[0].legend()
        ax[1].legend()

    plt.suptitle(title, size=24)
    fig.subplots_adjust(top=0.2)
    fig.tight_layout()
    if return_wandb_image:
        return wandb.Image(fig)
    else:
        return fig
