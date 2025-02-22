#from lightgbm import train
import numpy as np
import torch
import pandas as pd 
from sklearn.model_selection import train_test_split
import os
import h5py
from pathlib import Path
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


n=300
d=212
train_size = 0.8
batch_size = 32

data_dir = Path(r"../../data/VAN0046-LK-3-45-IMS_lipids_neg_roi=#1_mz=fix")
# data_dir2_5 = Path(r"data/negative/VAN0005-RK-2-5-IMS_lipids_neg_roi=#0_mz=fix")
# data_dir1_31 =  Path(r"data/negative/VAN0042-RK-1-31-IMS_lipids_neg_roi=#1_mz=fix")
# data_dir1_35 = Path(r"data/negative/VAN0049-RK-1-35-IMS_lipids_neg_roi=#1_mz=fix")
# data_dir1_41 = Path(r"data/negative/VAN0063-RK-1-41-IMS_lipids_neg_roi=#1_mz=fix")

#data_dir = data_dir1_35

def reshape_array(y_data, image_shape, pixel_index, fill_value=np.nan):
    """
    Reshape 1D data to 2D heatmap

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

def load_normalization_data(data_dir: Path, norm = "0-95% TIC"):
    """Load normalization data."""
    filename = data_dir / f"normalizations-raw.h5"
    assert filename.exists(), f"File {filename} does not exist"
    with h5py.File(filename, mode="r") as f_ptr:
        normalization = f_ptr[f"Normalization/{norm}/normalization"][:]
    # in order to keep intensity of ion images more-or-less preserved, you must
    # rescale the normalization by the median value of the norm
    return normalization / np.median(normalization)

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


def get_centroids_filenames(data_dir: Path):
    """Get list of filenames that correspond to centroid files."""
    files = []
    for path in data_dir.iterdir():
        if path.suffix == ".h5":
            with h5py.File(path, "r") as f_ptr:
                if "Array" in f_ptr.keys():
                    files.append(path.name)
    return files

def load_mask(data_dir: Path, name: str = "Glomerulus.h5"):
    """Load mask data."""
    filename = data_dir / name
    assert filename.exists(), f"File {filename} does not exist"
    with h5py.File(filename, mode="r") as f_ptr:
        mask = f_ptr["Mask/mask"][:]
    return mask

def get_mask_filenames(data_dir: Path):
    """Get list of filenames that correspond to mask files."""
    files = []
    for path in data_dir.iterdir():
        if path.suffix == ".h5":
            with h5py.File(path, "r") as f_ptr:
                if "Mask" in f_ptr.keys():
                    files.append(path.name)
    return files

def unshape_array(image: np.ndarray, pixel_index: np.ndarray):
    """Retrieve original vector of intensities from an image."""
    image_flat = image.reshape(-1)
    y_data = image_flat[pixel_index]
    return y_data

def load(data_dir:Path):
        # load data from this directory
        #data_dir = Path(r"data/VAN0046-LK-3-45-IMS_lipids_neg_roi=#1_mz=fix")
        #data_dir = Path(r"small_data")
        # load image metadata
        _, _, pixel_index = load_image_info(data_dir)
        
        # load centroids data
        centroids, _ = load_centroids(data_dir)
        # load mask data
        #glomeruls_mask = load_mask(data_dir)
        #glomeruls_mask = unshape_array(glomeruls_mask, pixel_index)

        _, image_shape, _ = load_image_info(data_dir)
        mask , mask_to_name  = compute_mask(image_shape)
        mask = unshape_array(mask, pixel_index)

        return centroids,mask,mask_to_name, pixel_index
def load_shape_norm_mzs():
    #data_dir = Path(r"data/VAN0046-LK-3-45-IMS_lipids_neg_roi=#1_mz=fix")
    # load image metadata
    _, image_shape, _ = load_image_info(data_dir)
    # load normalization data
    norm = load_normalization_data(data_dir)
    # load centroids data
    _, mzs = load_centroids(data_dir)
    return (image_shape,norm,mzs)

def load_mask_data(dataset_directory, targetROI, image_shape):
    """ Load 2D mask array: each pixel is assigned a score indicating its probability of belonging to a given region of interest (ROI).
    The renal tissue functional units are possible ROIs: Glomerulus, Proximal_Tubule, Distal_Tubule, Collecting_Duct, Thick_Ascending_Limb. """
    filename = os.path.join(dataset_directory, targetROI + '.h5')
    file_h5py = h5py.File(filename, 'r+')
    group_mask = file_h5py['Mask']
    bool_mask = group_mask['mask'][()]
    group_metadata = file_h5py['Mask']['Metadata']
    prob_mask = group_metadata['scores']
    prob_mask = np.nan_to_num(np.array(prob_mask, dtype='float32'))
    dataset_name = filename[filename.find('VAN'):filename.find('-IMS')]
    return dataset_name, bool_mask, prob_mask





def return_data():
    centroids, glomeruls_mask, pixel_index = load()
    mask = unshape_array(glomeruls_mask, pixel_index)

    x_data = np.hstack((centroids, mask[:,np.newaxis]))
    x_data = torch.tensor(x_data)
    train_set, test_set = torch.utils.data.random_split(x_data, [train_size, 1-train_size],generator=torch.Generator().manual_seed(42))
    train_set = train_set[:]
    test_set = test_set[:]
    
    train_loader = torch.utils.data.DataLoader([ [train_set[i,np.newaxis,:-1], train_set[i,-1]] for i in range(len(train_set))], 
                                            shuffle=True, 
                                            batch_size=batch_size,
                                            pin_memory=True,
                                            drop_last=True)
    
    test_loader = torch.utils.data.DataLoader([ [test_set[i,np.newaxis,:-1], test_set[i,-1]] for i in range(len(test_set))], 
                                                shuffle= False, 
                                                batch_size=batch_size,
                                                pin_memory=True,
                                                drop_last=True)
    return (train_loader , test_loader)

def return_train_val_test():
    centroids, glomeruls_mask, pixel_index = load()
    mask = unshape_array(glomeruls_mask, pixel_index)

    x_data = np.hstack((centroids, mask[:,np.newaxis]))
    x_data = torch.tensor(x_data)
    train_set, test_set = torch.utils.data.random_split(x_data, [0.7,0.1,0.2],generator=torch.Generator().manual_seed(42))
    train_set = train_set[:]
    test_set = test_set[:]
    
    train_loader = torch.utils.data.DataLoader([ [train_set[i,np.newaxis,:-1], train_set[i,-1]] for i in range(len(train_set))], 
                                            shuffle=True, 
                                            batch_size=batch_size,
                                            pin_memory=True,
                                            drop_last=True)
    
    test_loader = torch.utils.data.DataLoader([ [test_set[i,np.newaxis,:-1], test_set[i,-1]] for i in range(len(test_set))], 
                                                shuffle= False, 
                                                batch_size=batch_size,
                                                pin_memory=True,
                                                drop_last=True)
    return (train_loader , test_loader)


def return_synthetic_data():
    batch_size=10
    a = np.zeros((10,10,5))
    b = np.zeros((10,10,5))

    scale1 = np.array([130,100,12,20,4])
    scale2 = np.array([8,4,30,120,98])
    a[2:6,2:6,:] = 3 * scale1
    b[4:8,4:8,:] = 1 * scale2
    c = a+b
    val = c.reshape(-1, c.shape[2])[:,np.newaxis]
    mask = np.ones(val.shape[0])
    loader = torch.utils.data.DataLoader([ [val[i,:].float(), mask[i]] for i in range(val.shape[0])], 
                                            shuffle=False, 
                                            batch_size=batch_size,
                                            pin_memory=True,
                                            drop_last=True)
    return loader
    

def full_index_normalized_data():
    centroids, glomeruls_mask, pixel_index = load()
    _, image_shape, _ = load_image_info(data_dir)

    df = pd.DataFrame(centroids)
    df['index'] = pixel_index
    #mask = glomeruls_mask
    mask , mask_to_name  = compute_mask(image_shape)   #mask_to_name : Dictionary from the coding Label to it's ROI name
    df['mask'] = unshape_array(mask, pixel_index)

    train = df.sample(frac=0.8,random_state=42)
    val_train = train.drop(['index','mask'], axis=1)
    
    scaler = StandardScaler()
    scaler.fit(val_train)
    val = df.drop(['index','mask'], axis=1)
    val = scaler.transform(val)
    
    val = np.array(val)[:,np.newaxis]
    index = np.array(df['index'])
    mask = np.array(df['mask'])
    loader = torch.utils.data.DataLoader([ [val[i], mask[i], index[i]] for i in range(val.shape[0])], 
                                        shuffle=False, 
                                        batch_size=batch_size,
                                        pin_memory=True,
                                        drop_last=True)
    return loader, mask_to_name

def compute_mask(image_shape):
    """
    Compute the mask value for each pixel

    Parameters
    ----------
    image_shape: tuple
        final shape of the image

    Returns
    -------
    im_array: np.array
        reshaped heatmap of shape `image_shape`
    mask_to_name : dictionary
        Dictionary from the coding Label to it's ROI name
    """
    dir_ROI = ['other', 'Glomerulus', 'Proximal_Tubule', 'Distal_Tubule', 'Collecting_Duct', 'Thick_Ascending_Limb']
    target_ROI_arr = ['other', 'Gl', 'Pr', 'Di', 'Co', 'Th']
    mask_to_name = zip(range(0,len(target_ROI_arr)+1),target_ROI_arr)
    mask_to_name = dict(mask_to_name)

    acc_mask = np.zeros(np.concatenate([[len(target_ROI_arr)],image_shape]))
    for i in mask_to_name.keys():
        if i!= 0 :
            _, _ , prob_mask = load_mask_data(data_dir,dir_ROI[i], image_shape)
            quant = np.quantile(prob_mask[prob_mask>0],q=0.75)
            acc_mask[i-1,:,:] = np.where(prob_mask>=quant, i , 0)
    final_mask = np.zeros(image_shape) ## Ignore pixels with double masks
    for i in range(acc_mask.shape[1]):
        for j in range(acc_mask.shape[2]):
            r = acc_mask[:,i,j]
            r = r[r!=0]
            if len(r) == 1 :
                final_mask[i,j] = r
            else :
                final_mask[i,j] = 0
    return final_mask, mask_to_name
        


    


def return_normalized_data():
    recompute = True

    if not recompute:
        with open('saved_data/normalized_kidney.npy', 'rb') as f:
            print('predefined normalized kidney data')
            loaders = np.load(f,allow_pickle=True)
    else :
        centroids, glomeruls_mask, pixel_index = load()
        df = pd.DataFrame(centroids)
        df['index'] = pixel_index
        df['mask'] = unshape_array(glomeruls_mask, pixel_index)

        train = df.sample(frac=0.8,random_state=42)
        test = df.drop(train.index)


        val_train = train.drop(['index','mask'], axis=1)
        val_test = test.drop(['index','mask'], axis=1)

        train = pd.DataFrame(train)
        test = pd.DataFrame(test) 
        
        scaler = StandardScaler()
        scaler.fit(val_train)
        val_train = scaler.transform(val_train)
        val_test = scaler.transform(val_test)

        val=[val_train,val_test]
        d = [train,test]
        loaders = []
        for i in range(len(val)) :
            v= val[i]
            df = d[i]
            v = np.array(v)[:,np.newaxis]
            
            #index = np.array(df['index'])
            mask = np.array(df['mask'])
            loader = torch.utils.data.DataLoader([ [v[j],mask[j]] for j in range(v.shape[0])], 
                                                shuffle=False, 
                                                num_workers=3,
                                                batch_size=batch_size,
                                                pin_memory=True,
                                                drop_last=True)
            loaders.append(loader)
        with open('saved_data/normalized_kidney.npy', 'wb') as f:
            np.save(f, loaders)
        #(Train loader, test loader)
    return loaders

def return_normalized_unloader_data():
    centroids, glomeruls_mask, pixel_index = load()
    _, image_shape, _ = load_image_info(data_dir)

    df = pd.DataFrame(centroids)
    df['index'] = pixel_index
    
    mask , mask_to_name  = compute_mask(image_shape)   #mask_to_name : Dictionary from the coding Label to it's ROI name
    df['mask'] = unshape_array(mask, pixel_index)

    train = df.sample(frac=0.8,random_state=42)
    test = df.drop(train.index)


    val_train = train.drop(['index','mask'], axis=1)
    val_test = test.drop(['index','mask'], axis=1)

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    #total = df.drop(['mask'])

    scaler = StandardScaler()
    scaler.fit(val_train)
    val_train = scaler.transform(val_train)
    val_test = scaler.transform(val_test)

    total = scaler.transform(df.drop(['index','mask'],axis = 1))
    label_total = df['mask']

    #val=[val_train,val_test]
  
    return val_train, val_test,total, label_total,mask_to_name




    

    
    

    








