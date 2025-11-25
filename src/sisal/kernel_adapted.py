"""Module for kernel adapted density estimation and plotting."""

import matplotlib.pyplot as plt
import numpy as np


def kernel_adapated(full_latent: np.ndarray, vars: np.ndarray) -> np.ndarray:
    X, Y = np.mgrid[-4:4:200j, -4:4:200j]
    grid = np.vstack([X.ravel(), Y.ravel()])
    im_size = len(X[0])

    n = len(full_latent)
    h = 4 * n ** (-1 / 6)

    image = np.zeros(len(grid[0]))

    n = len(full_latent)
    for i in range(n):
        image = image + np.exp(
            -0.5
            * ((grid[0] - full_latent[i, 1]) ** 2 / (vars[i, 0]) + (grid[1] - full_latent[i, 2]) ** 2 / (vars[i, 1]))
            / h**2
            - 0.5 * (np.log(vars[i, 0]) + np.log(vars[i, 1]))
        )

    image = image / (2 * np.pi * n * h**2)
    image_r = image.reshape((im_size, im_size), order="C")
    image_r = np.rot90(image_r)
    return image_r


def plot_kernel_adapted(ax: plt.Axes, fig: plt.Figure, image: np.ndarray) -> None:
    # np.rot90(image_r)
    # fig,ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(image, extent=[-4, 4, -4, 4], cmap="jet")
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    p = ax.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p[0], 0.95, p[2] - p[0], 0.05])
    cb = plt.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cb.ax.tick_params(labelsize=20)
