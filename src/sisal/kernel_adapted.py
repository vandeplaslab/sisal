"""Module for kernel adapted density estimation and plotting."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper

    def prange(*args, **kwargs):
        return range(*args, **kwargs)


def kernel_adapated(full_latent: np.ndarray, vars: np.ndarray) -> np.ndarray:
    X, Y = np.mgrid[-4:4:200j, -4:4:200j]
    grid = np.vstack([X.ravel(), Y.ravel()])
    im_size = len(X[0])

    n = len(full_latent)
    h = 4 * n ** (-1 / 6)

    image = np.zeros(len(grid[0]))

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


def kernel_adapted_opt(full_latent: np.ndarray, vars: np.ndarray) -> np.ndarray:
    if not HAS_NUMBA:
        return kernel_adapated(full_latent, vars)

    X, Y = np.mgrid[-4:4:200j, -4:4:200j]
    im_size = X.shape[0]

    grid_x = X.ravel()
    grid_y = Y.ravel()

    n = len(full_latent)
    h = 4 * n ** (-1 / 6)

    # We pass the flattened arrays to the JIT function
    image = _compute_density_corrected(
        grid_x, grid_y, full_latent[:, 1], full_latent[:, 2], vars[:, 0], vars[:, 1], h, n
    )

    image = image / (2 * np.pi * n * h**2)
    image_r = image.reshape((im_size, im_size), order="C")
    image_r = np.rot90(image_r)

    return image_r


@njit(parallel=True, fastmath=True)
def _compute_density_corrected(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    latents_1: np.ndarray,
    latents_2: np.ndarray,
    vars_0: np.ndarray,
    vars_1: np.ndarray,
    h: float,
    n: int,
) -> np.ndarray:
    n_pixels = grid_x.shape[0]
    image = np.zeros(n_pixels)

    # Pre-calculate the log part.
    log_terms = 0.5 * (np.log(vars_0) + np.log(vars_1))

    # Pre-calculate constants
    h_sq = h**2

    for p in prange(n_pixels):
        gx = grid_x[p]
        gy = grid_y[p]
        sum_val = 0.0

        for i in range(n):
            # Compute Distance Squared
            dist_sq = (gx - latents_1[i]) ** 2 / vars_0[i] + (gy - latents_2[i]) ** 2 / vars_1[i]
            # Apply -0.5 and divide by h^2 (First Term)
            term_1 = -0.5 * dist_sq / h_sq
            # Subtract the log term (Second Term)
            exponent = term_1 - log_terms[i]
            sum_val += np.exp(exponent)
        image[p] = sum_val
    return image


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
