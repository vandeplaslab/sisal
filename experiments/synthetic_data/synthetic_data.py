import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

# import torch
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# eps_scale = 2
eps_scale = 10
coeff_scale = 2
coeff_loc = 2

## after normalization
eps_scale = 10
coeff_scale = 500
coeff_loc = 500


cmap_spa = "cmr.pride"


class Shape:
    def __init__(self, cx, cy):
        self.c = np.array([cx, cy])

    ## Returns True if the pixel(x,y) is in the shape
    def is_in(x, y):
        raise NotImplementedError


class Circle(Shape):
    ## Radius center at (x,y) and radius r
    def __init__(self, cx, cy, r):
        super().__init__(cx, cy)
        self.r = r

    def is_in(self, x, y):
        pos = np.array([x, y])
        return np.linalg.norm(self.c - pos) <= self.r


class Square(Shape):
    def __init__(self, cx, cy, l):
        super().__init__(cx, cy)
        self.l = l

    def is_in(self, x, y):
        return np.abs(x - self.c[0]) <= self.l / 2 and np.abs(y - self.c[1]) <= self.l / 2


class Triangle(Shape):
    ## Triangle of center (cx, cy), width w and height h
    #         B
    #        / \
    #       /   \
    #      /     \
    #     /   c   \
    #    /         \
    #  A ----------- C
    def __init__(self, cx, cy, w, h):
        super().__init__(cx, cy)
        self.A = self.c + np.array([-w, -h]) / 2
        self.B = self.c + np.array([w, -h]) / 2
        self.C = self.c + np.array([0, h]) / 2
        self.base_area = self.area(self.A, self.B, self.C)

    ## Area of triangle given by points a1,a2,a3
    def area(self, a1, a2, a3):
        # [ x1(y2 – y3) + x2(y3 – y1) + x3(y1-y2)]/2
        return np.abs(a1[0] * (a2[1] - a3[1]) + a2[0] * (a3[1] - a1[1]) + a3[0] * (a1[1] - a2[1])) / 2

    def is_in(self, x, y):
        p = np.array([x, y])
        area1 = self.area(self.A, p, self.C)
        area2 = self.area(self.A, p, self.B)
        area3 = self.area(self.B, p, self.C)

        return self.base_area == (area1 + area2 + area3)


def create_image(eps_scale, coeff_scale, coeff_loc):
    image_shape = np.array([358, 666])
    # image_total = np.zeros(image_shape)
    image_circle = np.zeros(image_shape)
    image_triangle = np.zeros(image_shape)
    image_square = np.zeros(image_shape)
    mask = np.zeros(image_shape)
    c = Circle(120, 200, 100)
    t = Triangle(250, 350, 200, 300)
    s = Square(130, 350, 200)

    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            # print(t.is_in(i,j))
            in_c = c.is_in(i, j)
            in_t = t.is_in(i, j)
            in_s = s.is_in(i, j)
            if in_c:
                image_circle[i, j] = np.random.normal(loc=coeff_loc, scale=coeff_scale)
            if in_t:
                image_triangle[i, j] = np.random.normal(loc=coeff_loc, scale=coeff_scale)
            if in_s:
                image_square[i, j] = np.random.normal(loc=coeff_loc, scale=coeff_scale)
            mask[i, j] = get_mask([in_s, in_t, in_c])

    return image_circle, image_triangle, image_square, mask


def get_colormap():
    # Import the 'cmr.pride' colormap
    # pride_cmap = plt.get_cmap('cmr.pride')
    n_col = 2**3
    cmap_spa = "cmr.pride"
    new_colors = cmr.take_cmap_colors(cmap_spa, n_col, return_fmt="hex")
    new_colors[-1] = [0.0, 0.5, 1.0, 1.0]  # Set the last color to blue
    modified_pride_cmap = LinearSegmentedColormap.from_list("pride", new_colors, N=n_col)

    return modified_pride_cmap


def create_image_new(eps_scale, coeff_scale, coeff_loc):
    image_shape = np.array([358, 666])
    image_circle = np.zeros(image_shape)
    image_triangle = np.zeros(image_shape)
    image_square = np.zeros(image_shape)
    mask = np.zeros(image_shape)
    c = Circle(120, 200, 100)
    t = Triangle(250, 350, 200, 300)
    s = Square(130, 350, 200)

    overlapp = np.zeros(image_shape)  ## Remover issues with 1/counter = 0 by adding ones

    np.random.seed(seed=3868)
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            # print(t.is_in(i,j))
            in_c = c.is_in(i, j)
            in_t = t.is_in(i, j)
            in_s = s.is_in(i, j)
            if in_c:
                image_circle[i, j] = np.abs(np.random.normal(loc=coeff_loc, scale=coeff_scale))
            if in_t:
                image_triangle[i, j] = np.abs(np.random.normal(loc=coeff_loc, scale=coeff_scale))
            if in_s:
                image_square[i, j] = np.abs(np.random.normal(loc=coeff_loc, scale=coeff_scale))

            count = int(in_c) + int(in_t) + int(in_s)
            if count == 0 or count == 1:
                overlapp[i, j] = 1
            else:
                overlapp[i, j] = 1 / count
            mask[i, j] = get_mask([in_s, in_t, in_c])

    return image_circle, image_triangle, image_square, mask, overlapp


def plot_spatial(image, mask, eps_scale=eps_scale, coeff_scale=coeff_scale, coeff_loc=coeff_loc):
    _, ax = plt.subplots(2, 1, figsize=(30, 7))
    ax[0].imshow(mask, cmap=cmap_spa)
    ax[1].imshow(image)
    ax[0].title.set_text("Mask")
    ax[1].title.set_text(rf"$\epsilon \sim N(0,{eps_scale}^2), c,t,s \sim N({coeff_loc},{coeff_scale}^2)$")
    plt.savefig("plots/synthetic_data/complete/spatial_pattern.png", bbox_inches="tight", dpi=300)


def plot_spatial_separate(mask, image_circle, image_triangle, image_square):
    _, ax = plt.subplots(ncols=4, figsize=(10, 7))
    ax[0].tick_params(axis="x", labelsize=10)
    ax[0].tick_params(axis="y", labelsize=10)
    ax[0].imshow(mask, cmap=get_colormap())
    # plt.savefig('plots/synthetic_data/complete/spatial_pattern.png',bbox_inches='tight',dpi=300)

    # _,ax = plt.subplots(figsize =(15,15))
    images = [image_circle, image_triangle, image_square]
    for i, image in enumerate(images):
        # _,ax = plt.subplots(figsize=(8,5))
        ax[i + 1].tick_params(axis="x", labelsize=25)
        ax[i + 1].tick_params(axis="y", labelsize=25)
        #####
        image = np.ma.masked_where(image == 0, image)
        cmap = plt.cm.get_cmap().copy()
        cmap.set_bad(color="white")
        ####
        ax[i + 1].imshow(image, cmap)
        ax[i + 1].set_xticks([])
        ax[i + 1].set_yticks([])
        # plt.savefig('plots/synthetic_data/complete/coefficients_{}.png'.format(image_title[i]),bbox_inches='tight',dpi=300)


def plot_synthetic():
    image_circle, image_triangle, image_square, mask, overlapp = create_image_new(
        eps_scale=eps_scale, coeff_scale=coeff_scale, coeff_loc=coeff_loc
    )
    image_circle + image_triangle + image_square
    in_size = 212

    # plot_spatial(image_total,mask)
    plot_spatial_separate(mask, image_circle, image_triangle, image_square)

    spectrums, spectrum_names = create_spectrum(in_size)
    plot_spectral_separate(spectrums, spectrum_names)


# images = [circle, triangle,square]
# def plot_spatial_separate_past(eps_scale=eps_scale , coeff_scale=coeff_scale , coeff_loc=coeff_loc ):
#     image_circle, image_triangle, image_square,mask = create_image(eps_scale,coeff_scale,coeff_loc)
#     images = [image_circle, image_triangle, image_square]
#     _, ax = plt.subplots(2,2,figsize=(8, 5))
#     #print(len(images))

#     ax[0,0].imshow(mask,cmap=cmap_spa)
#     ax[0,0].title.set_text(r"Param $\epsilon \sim N(0,{}^2), c,t,s \sim N({},{}^2)$".format(eps_scale,coeff_loc,coeff_scale))
#     ax[1,0].imshow(images[0])
#     ax[1,0].title.set_text('Circle coeff')
#     ax[0,1].imshow(images[1])
#     ax[0,1].title.set_text('Triangle coeff')
#     ax[1,1].imshow(images[2])
#     ax[1,1].title.set_text('Square coeff')
# plt.title(r"$\epsilon \sim N(0,{}^2), c,t,s \sim N({},{}^2)$".format(eps_scale,coeff_loc,coeff_scale))

# ax[1].title.set_text(r"$\epsilon \sim N(0,{}^2), c,t,s \sim N({},{}^2)$".format(eps_scale,coeff_loc,coeff_scale))
# plt.savefig('plots/synthetic_data/separate_coeff.png',bbox_inches='tight',dpi=300)


def plot_spatial_separate2d(eps_scale=eps_scale, coeff_scale=coeff_scale, coeff_loc=coeff_loc):
    _, ax = plt.subplots(nrows=2, figsize=(10, 10))

    image_circle, image_triangle, image_square, mask = create_image(eps_scale, coeff_scale, coeff_loc)
    image_tot = image_circle + image_triangle + image_square

    ax[0].imshow(mask, cmap=cmap_spa)
    ax[0].title.set_text(rf"Param $\epsilon \sim N(0,{eps_scale}^2), c,t,s \sim N({coeff_loc},{coeff_scale}^2)$")
    ax[1].imshow(image_tot)
    ax[1].title.set_text("Coefficients")

    plt.savefig("plots/synthetic_data/mask_coeff.png", bbox_inches="tight")
    plt.savefig("plots/synthetic_data/mask_coeff.pdf", bbox_inches="tight")


def plot_mask(eps_scale=eps_scale, coeff_scale=coeff_scale, coeff_loc=coeff_loc):
    plt.figure(figsize=(10, 10))
    image_circle, image_triangle, image_square, mask = create_image(eps_scale, coeff_scale, coeff_loc)
    plt.imshow(mask, cmap=cmap_spa)
    plt.title(rf"Param $\epsilon \sim N(0,{eps_scale}^2), c,t,s \sim N({coeff_loc},{coeff_scale}^2)$")
    plt.savefig("plots/synthetic_data/mask.png", bbox_inches="tight")


def plot_spectral(s_col):
    len_s_col = len(s_col)
    n_col = 2**len_s_col
    colors = cmr.take_cmap_colors(cmap_spa, n_col, return_fmt="hex")
    col_dict = dict(zip(range(n_col), colors))

    _, ax = plt.subplots(len_s_col, 1, figsize=(15, 10))
    mask_to_name = mask_to_name_synthetic()

    names = []
    index_col = []
    for i in range(len_s_col):
        binary = np.zeros(len_s_col)
        binary[len_s_col - i - 1] = 1
        index = get_mask(binary)
        index_col.append(index)
        names.append(mask_to_name[index])

    short_to_long = {"c": "Circle", "t": "Triangle", "s": "Square"}
    for i, s in enumerate(s_col):
        z_size = len(s)
        x_pos = np.arange(z_size)

        ax[i].title.set_text(short_to_long[names[i]])
        _, stemlines1, _ = ax[i].stem(x_pos, s, col_dict[index_col[i]], markerfmt=" ")
        plt.setp(stemlines1, "linewidth", 3)

    plt.savefig("plots/synthetic_data/spectral_pattern.png", bbox_inches="tight", dpi=300)


def plot_spectral_separate(s_col, spectrum_names):
    n_col = 2**3
    cmap_spa = "cmr.pride"
    new_colors = cmr.take_cmap_colors(cmap_spa, n_col, return_fmt="hex")
    new_colors[-1] = [0.0, 0.5, 1.0, 1.0]  # Set the last color to blue
    col_dict = dict(zip(range(len(new_colors)), new_colors))
    len_s_col = len(s_col)

    mask_to_name = mask_to_name_synthetic()

    names = []
    index_col = []
    for i in range(len_s_col):
        binary = np.zeros(len_s_col)
        binary[len_s_col - i - 1] = 1
        index = get_mask(binary)
        index_col.append(index)
        names.append(mask_to_name[index])

    _, axs = plt.subplots(figsize=(10, 5), nrows=3)
    for i, s in enumerate(s_col):
        axs[i].tick_params(axis="x", labelsize=10)
        axs[i].tick_params(axis="y", labelsize=10)
        z_size = len(s)
        x_pos = np.arange(z_size)

        _, stemlines1, _ = axs[i].stem(x_pos, s, col_dict[index_col[i]], markerfmt=" ")


def mask_to_name_synthetic():
    names = np.array(["s", "t", "c"])
    ordered_value = []
    order_name = []
    for i in range(2 ** len(names)):
        s_binary = bin(i)[2:].zfill(3)
        v = [int(i) for i in s_binary]
        ordered_value.append(get_mask(v))
        name = "+".join(names[np.array(v) == 1])
        order_name.append(name)

    in_to_name = dict(zip(ordered_value, order_name))
    in_to_name[0] = "noise"
    return in_to_name


def get_mask(is_in_array):
    return sum(val * (2**idx) for idx, val in enumerate(reversed(is_in_array)))


def plot_all():
    z_size = 212

    image_circle, image_triangle, image_square, mask = create_image(
        eps_scale=eps_scale, coeff_scale=coeff_scale, coeff_loc=coeff_loc
    )
    image_circle + image_triangle + image_square

    # plot_spatial(image_total,mask)
    r1 = np.random.RandomState(1920)
    s1 = r1.uniform(0, 1000, z_size)
    r2 = np.random.RandomState(4174)
    s2 = r2.uniform(0, 1000, z_size)
    # plot_spectral(s1,s2)

    i = 10
    # PATH = 'plots/synthetic_data/spectral_cut/'
    for i in range(0, z_size, z_size // 10):
        temp_c = s1[i] * image_circle
        temp_t = s2[i] * image_triangle

        temp_f = temp_c + temp_t + np.random.normal(loc=0, scale=eps_scale, size=temp_c.shape)

        full_title = f"full_z{i}.pdf"
        plt.figure(full_title)
        plt.imshow(temp_f)
        plt.close()


def create_spectrum(in_size):
    r1 = np.random.RandomState(1920)
    s1 = r1.uniform(0, 1000, in_size)
    r2 = np.random.RandomState(4174)
    s2 = r2.uniform(0, 1000, in_size)
    r3 = np.random.RandomState(3264)
    s3 = r3.uniform(0, 1000, in_size)

    ## Normalise the 3 signals :
    s1 = s1 / np.linalg.norm(s1)
    s2 = s2 / np.linalg.norm(s2)
    s3 = s3 / np.linalg.norm(s3)

    spectrum_names = ["circle", "triangle", "square"]
    return [s1, s2, s3], spectrum_names


def return_synthetic_data():
    in_size = 212

    image_circle, image_triangle, image_square, mask, overlapp = create_image_new(
        eps_scale=eps_scale, coeff_scale=coeff_scale, coeff_loc=coeff_loc
    )

    s, _ = create_spectrum(in_size)

    C = overlapp.flatten().reshape(-1, 1)
    data = C * (
        np.outer(image_circle.flatten(), s[0])
        + np.outer(image_triangle.flatten(), s[1])
        + np.outer(image_square.flatten(), s[2])
    )
    # Add noise

    r = np.random.RandomState(8970)
    centroids = data + r.normal(scale=eps_scale, size=data.shape)

    # The alpha value (SNR) of the spectrum
    # COMPUTE the SNR named alpha which is the norm of the spectrum signal
    SNR = np.linalg.norm(data, axis=1)**2 / (eps_scale**2 * data.shape[1])

    return centroids, SNR, mask.flatten(), mask_to_name_synthetic()
