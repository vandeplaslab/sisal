import numpy as np
from pathlib import Path
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import PIL

DATA_DIR = Path(r"../../data/")

def load_data(im_num: str = "0012"):
    dir = DATA_DIR / "salient/HS-SOD/hyperspectral/"
    name = im_num + ".mat"
    with h5py.File(dir / name, mode="r") as f_ptr:
        dataset = np.array(f_ptr["hypercube"])
        dataset = dataset.transpose((2, 1, 0))
    return dataset

def spectral_range():
    """
    Return the spectral values of the features

    Returns
    -------
    nm_values: np.array
        array of the spectral values in nm
    """
    nm_values = np.array([f"{380 + 5 * i}" for i in range(81)])
    return nm_values

def return_rgb_image(im_num: str = "0012"):
    """
    Return an RBG image example of the Salient dataset
    """
    dir = DATA_DIR / "salient/HS-SOD/color"
    name = im_num + ".jpg"
    im = PIL.Image.open(dir / name)
    return im

def reshape_array(y_data, image_shape, pixel_index, fill_value=np.nan):
    """
    Reshape 1D data to 2D heatmap.

    Parameters
    ----------
    y_data: np.array / list
        1D array of values to be reshaped
    image_shape: tuple
        final shape of the image
    pixel_index: np.array
        array containing positions where pixels should be placed, considering missing values -
        e.g. not acquired pixels
    fill_zeros: bool
        determines whether missing values should be filled with NaNs or 0s
    check_orientation : bool
        if True, image will be rotated to ensure the width is larger than the height
    fill_value : float, optional
        if value is provided, it will be used to fill-in the values

    Returns
    -------
    im_array: np.array
        reshaped heatmap of shape `image_shape`
    """
    if isinstance(y_data, np.matrix):
        y_data = np.asarray(y_data).flatten()
    y_data = np.asarray(y_data)
    dtype = np.float32 if isinstance(fill_value, float) else y_data.dtype

    image_n_pixels = np.prod(image_shape)
    im_array = np.full(image_n_pixels, dtype=dtype, fill_value=fill_value)
    im_array[pixel_index] = y_data

    # reshape data
    im_array = np.reshape(im_array, image_shape)
    return im_array

def index_to_image_pos() -> dict:
    """Map pixel index to image position."""
    _,pixel_index,image_shape = return_data_salient()

    reshape_pixel = reshape_array(pixel_index, image_shape, pixel_index)
    index_to_pos = {}
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            if ~np.isnan(reshape_pixel[i, j]):
                index_to_pos[int(reshape_pixel[i, j])] = np.array([i, j])
    return index_to_pos

def return_data_salient(im_num: str = "0012"):
    print(f"Loading image {im_num}")
    dataset = load_data(im_num)

    image_shape = dataset.shape[:2]

    # reshape the data on the pixel side
    data_flat = dataset.reshape((-1, dataset.shape[2]))
    pixel_index = np.arange(data_flat.shape[0])

    df = pd.DataFrame(data_flat)
    df["index"] = pixel_index
    val = df.drop(["index"], axis=1)

    return val.to_numpy(), np.array(df["index"]), image_shape

def variance_latent_traversal_hyperspectral(combined_p, nm_values, d):

    ## Subset of x_ticks that are going to be ploted
    #subset = np.arange(0, combined_p[0].shape[1], step=4)
    sign_traversal = np.zeros((len(combined_p), combined_p[0].shape[1]))
    for i, p_steps in enumerate(combined_p):
        ## If this difference is positive it means the signal was increasing
        sign_traversal[i, :] = (p_steps[-1, :] - p_steps[0, :] >= 0) * 2 - 1  ## Casting values to the range [-1,1]

    ## For interpretation note that some traversal increase the value and some decrease it
    sign_traversal = np.sum(sign_traversal, axis=0) >= 0

    ## Plot of the stacked traversal
    data_var = {}
    for i, p_steps in enumerate(combined_p):
        data_var[f"var_{i}"] = np.var(p_steps, axis=0)

    sum_var = np.sum(np.array(list(data_var.values())), axis=0)
    col_data = pd.DataFrame({"nm_values": nm_values, "variance": sum_var, "increase": sign_traversal})
    col_data = col_data.sort_values(by="variance", ascending=False)
    ## Plot of the variance in order :
    col_data = col_data.sort_values(by="variance", ascending=False)

    _f, ax = plt.subplots(figsize=(15, 5))

    subset = np.arange(0, d, step=2)
    ax.set_xticks(subset)
    ticks_labels_nm = np.array(col_data["nm_values"])
    ticks_labels = np.array([ticks_labels_nm[i] for i in subset])
    ax.set_xticklabels(ticks_labels, rotation=45, ha="right")



    color_bars = {1: "#050533", 0: "#E34234"}  # Darkblue, red
    color_bars = {1: "#fb6f92", 0: "#8d99ae"}  # Darkblue, red
    color_bars = {1: "#fe6d73", 0: "#17c3b2"}  # Darkblue, red
    label_bars = {1: "L-to-R increasing", 0: "L-to-R decreasing"}

    axins = ax.inset_axes((0.3, 0.6, 0.4, 0.3))
    ax.tick_params(axis="y", labelsize=15)
    axins.tick_params(axis="y", labelsize=15)
    axins.tick_params(axis="x", labelsize=15)

    cutx = 14
    X_global = np.arange(len(sum_var))
    X1 = X_global[:cutx]
    X2 = X_global[cutx:]
    Y1 = col_data["variance"][:cutx]
    Y2 = col_data["variance"][cutx:]

    col_data_sub = col_data[:cutx]["increase"]
    for l in np.unique(col_data_sub):
        ind = col_data_sub == l
        X = X1[ind]
        Y = Y1[ind]
        ax.scatter(X, Y, c=color_bars[l])
        axins.scatter(X, Y, c=color_bars[l], label=label_bars[l])

    subset = np.arange(0, cutx, step=1)
    ticks_labels_in = np.array([ticks_labels_nm[i] for i in subset])
    axins.set_xticks(subset)
    axins.set_xticklabels(ticks_labels_in, rotation=45, ha="right")
    axins.set_xlabel('nm', fontsize=15)
    
    # Plot the rest
    col_data_sub = col_data[cutx:]["increase"]
    for l in np.unique(col_data_sub):
        ind = col_data_sub == l
        X = X2[ind]
        Y = Y2[ind]
        ax.scatter(X, Y, c=color_bars[l], label=label_bars[l])

    mark_inset(ax, axins, loc1=1, loc2=3)
    ax.legend(prop={"size": 15})
    ax.set_xlabel('nm', fontsize=15)
    ax.set_ylabel('Variance', fontsize=15)