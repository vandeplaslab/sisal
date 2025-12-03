# Standard library imports
import os
from pathlib import Path

# Third party imports
import h5py
import numpy as np


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


def load_image_info(data_dir: Path):
    """Load image indexing information."""
    filename = data_dir / "image_info.npz"
    assert filename.exists(), f"File {filename} does not exist"
    with np.load(filename) as f_ptr:
        total_n_px = f_ptr["total_n_pixels"]
        image_shape = f_ptr["image_shape"]
        pixel_order = f_ptr["pixel_order"]
    return total_n_px, image_shape, pixel_order


def load_normalization_data(data_dir: Path, norm="0-95% TIC"):
    """Load normalization data."""
    filename = data_dir / "normalizations-raw.h5"
    assert filename.exists(), f"File {filename} does not exist"
    with h5py.File(filename, mode="r") as f_ptr:
        normalization = f_ptr[f"Normalization/{norm}/normalization"][:]
    # in order to keep intensity of ion images more-or-less preserved, you must
    # rescale the normalization by the median value of the norm
    return normalization / np.median(normalization)


def index_to_image_pos(data_dir: Path) -> dict:
    """Map pixel index to image position."""
    _, _, pixel_index = load_image_info(data_dir)
    image_shape, _, _ = load_shape_norm_mzs(data_dir)

    reshape_pixel = reshape_array(pixel_index, image_shape, pixel_index)
    index_to_pos = {}
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            if ~np.isnan(reshape_pixel[i, j]):
                index_to_pos[int(reshape_pixel[i, j])] = np.array([i, j])
    return index_to_pos


def load_centroids(data_dir: Path, name: str = "peaks_ppm=3.0_final_mzs_v1.h5"):
    """Load centroids data."""
    filename = data_dir / name
    assert filename.exists(), f"File {filename} does not exist"
    with h5py.File(filename, mode="r") as f_ptr:
        ds = f_ptr["Array/array"]
        array = np.zeros(ds.shape, dtype=ds.dtype)
        # because the array is chunked, it is much more efficient to read the data
        # in this way than reading it all at once
        for chunk in ds.iter_chunks():
            array[chunk] = ds[chunk][:]
        mzs = f_ptr["Array/xs"][:]
    return array, mzs


def load_mask(data_dir: Path, name: str = "Glomerulus.h5") -> np.ndarray:
    """Load mask data."""
    filename = data_dir / name
    assert filename.exists(), f"File {filename} does not exist"
    with h5py.File(filename, mode="r") as f_ptr:
        mask = f_ptr["Mask/mask"][:]
    return mask


def unshape_array(image: np.ndarray, pixel_index: np.ndarray) -> np.ndarray:
    """Retrieve original vector of intensities from an image."""
    image_flat = image.reshape(-1)
    y_data = image_flat[pixel_index]
    return y_data


def load(data_dir: Path) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    # load image metadata
    _, _, pixel_index = load_image_info(data_dir)
    # load centroids data
    centroids, mzs = load_centroids(data_dir)
    # load image shape
    _, image_shape, _ = load_image_info(data_dir)
    # load mask data
    mask, mask_to_name = compute_mask(data_dir, image_shape)
    mask = unshape_array(mask, pixel_index)
    return centroids, mask, mask_to_name, pixel_index, mzs


def load_shape_norm_mzs(data_dir: Path) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    # data_dir = Path(r"data/VAN0046-LK-3-45-IMS_lipids_neg_roi=#1_mz=fix")
    # load image metadata
    _, image_shape, _ = load_image_info(data_dir)
    # load normalization data
    norm = load_normalization_data(data_dir)
    # load centroids data
    _, mzs = load_centroids(data_dir)
    return image_shape, norm, mzs


def load_mask_data(data_dir: Path, targetROI: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Load 2D mask array: each pixel is assigned a score indicating its probability of belonging to a given region of interest (ROI).
    The renal tissue functional units are possible ROIs: Glomerulus, Proximal_Tubule, Distal_Tubule, Collecting_Duct, Thick_Ascending_Limb.
    """
    filename = os.path.join(data_dir, targetROI + ".h5")
    file_h5py = h5py.File(filename, "r+")
    group_mask = file_h5py["Mask"]
    bool_mask = group_mask["mask"][()]
    group_metadata = file_h5py["Mask"]["Metadata"]
    prob_mask = group_metadata["scores"]
    prob_mask = np.nan_to_num(np.array(prob_mask, dtype="float32"))
    dataset_name = filename[filename.find("VAN") : filename.find("-IMS")]
    return dataset_name, bool_mask, prob_mask


def compute_mask(data_dir: Path, image_shape: tuple) -> tuple[np.ndarray, dict]:
    """
    Compute the mask value for each pixel.

    Parameters
    ----------
    data_dir: Path
        directory where the data is stored
    image_shape: tuple
        final shape of the image

    Returns
    -------
    im_array: np.array
        reshaped heatmap of shape `image_shape`
    mask_to_name : dictionary
        Dictionary from the coding Label to it's ROI name
    """
    dir_ROI = ["other", "Glomerulus", "Proximal_Tubule", "Distal_Tubule", "Collecting_Duct", "Thick_Ascending_Limb"]
    target_ROI_arr = ["other", "Gl", "Pr", "Di", "Co", "Th"]
    mask_to_name = zip(range(0, len(target_ROI_arr) + 1), target_ROI_arr)
    mask_to_name = dict(mask_to_name)

    acc_mask = np.zeros(np.concatenate([[len(target_ROI_arr)], image_shape]))
    for i in mask_to_name:
        if i != 0:
            _, _, prob_mask = load_mask_data(data_dir, dir_ROI[i])
            quant = np.quantile(prob_mask[prob_mask > 0], q=0.75)
            acc_mask[i - 1, :, :] = np.where(prob_mask >= quant, i, 0)
    final_mask = np.zeros(image_shape)  ## Ignore pixels with double masks
    for i in range(acc_mask.shape[1]):
        for j in range(acc_mask.shape[2]):
            r = acc_mask[:, i, j]
            r = r[r != 0]
            if len(r) == 1:
                final_mask[i, j] = r
            else:
                final_mask[i, j] = 0
    return final_mask, mask_to_name
