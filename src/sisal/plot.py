from __future__ import annotations

from itertools import groupby
from pathlib import Path

import cmasher as cmr
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import path as m_path
from matplotlib.patches import Ellipse, Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy import stats

from sisal.utils import compute_latent, compute_loss, reparametrize

mpl.rcParams.update(mpl.rcParamsDefault)

class Plot:
    def __init__(
        self, path: Path, device: str, train_loader, test_loader, full_loader, output_dir: Path | None = None
    ) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loader = full_loader

        device = torch.device(device)
        model = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        self.model = model
        full_latent, vars, label, alpha = compute_latent(self.loader, model)

        self.full_latent = full_latent
        self.vars = vars
        self.label = label
        self.alpha = alpha

        # set output directory
        if output_dir is None:
            output_dir = Path(path).parent
        self.output_dir = Path(output_dir)

    def get_cov_ellipse(self, z_mean, z_var, nstd, color, **kwargs):
        # Width and height of ellipse to draw
        width, height = 2 * nstd * np.sqrt(z_var)
        return Ellipse(xy=z_mean, width=width, height=height, fill=False, color=color, alpha=1, linewidth=1, **kwargs)

    ## Add to ax the scatter points and the covariance matrix around for a proportion p
    def scatter_with_covar(self, ax, full_latent, vars, label, col_dict, mask_to_name):
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)

        if len(col_dict) == 1:
            for i in np.unique(label):
                mask = label == i
                ax.scatter(x=full_latent[mask, 0], y=full_latent[mask, 1], s=0.01, alpha=0.5)
            for i in range(vars.shape[0]):
                e = self.get_cov_ellipse(full_latent[i, :], vars[i, :], 1, color="royalblue")
                ax.add_artist(e)
        else:
            for i in np.unique(label):
                mask = label == i
                ax.scatter(
                    x=full_latent[mask, 0],
                    y=full_latent[mask, 1],
                    s=0.01,
                    alpha=0.8,
                    color=col_dict[i],
                    label=mask_to_name[i],
                )
            for i in range(vars.shape[0]):
                e = self.get_cov_ellipse(full_latent[i, :], vars[i, :], 1, color=col_dict[label[i]])
                ax.add_artist(e)
            ax.legend(markerscale=70, loc="upper right", fontsize="20")

    # p subsample proportion of full _latent
    def plot_latent_dim_with_var(self, mask_to_name, p=0.5, random_state_synthetic=1326):
        int(self.full_latent.shape[0] * p)
        col_dict = self.color_dict(n_col=len(np.unique(self.label)) + 1)

        if p != 1:
            r = np.random.RandomState(random_state_synthetic)
            sub_index = r.choice(self.full_latent.shape[0], int(self.full_latent.shape[0] * p), replace=False)
            full_latent = self.full_latent[sub_index]
            label = self.label[sub_index]
            vars = self.vars[sub_index]

        ## Colored version
        _, ax = plt.subplots(figsize=(10, 10))
        plt.xlim((-4.5, 4.5))
        plt.ylim((-4.5, 4.5))

        self.scatter_with_covar(ax, full_latent[:, 1:], vars, label, col_dict, mask_to_name)

        _, ax = plt.subplots(figsize=(10, 10))
        plt.xlim((-4.5, 4.5))
        plt.ylim((-4.5, 4.5))

        col_dict = {0: "blue"}
        self.scatter_with_covar(ax, full_latent[:, 1:], vars, label, col_dict, mask_to_name)
        plt.close()

    ## p:proportions of all points plotted
    # Plot for the synthetic data in which we know the scaling value (SNR)
    # Plot the scaling values in latent space
    def plot_latent_dim_coeff(self, p=1, random_state_synthetic=1326):
        n = self.full_latent.shape[0]
        alpha_label = self.alpha
        full_latent = self.full_latent

        if p != 1:
            r = np.random.RandomState(random_state_synthetic)
            sub_index = r.choice(n, int(n * p), replace=False)
            full_latent = self.full_latent[sub_index]
            alpha_label = alpha_label[sub_index]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)

        x = full_latent[:, 1]
        y = full_latent[:, 2]

        plt.xlim((np.min(x), np.max(x)))
        plt.ylim((np.min(y), np.max(y)))

        sc = ax.scatter(x=x, y=y, c=alpha_label, alpha=0.5, cmap="viridis")

        pos = ax.get_position().get_points().flatten()
        ax_cbar = fig.add_axes([pos[0] + 0.04, 0.84, (pos[2] - pos[0]) * 0.8, 0.02])
        cb = plt.colorbar(sc, cax=ax_cbar, orientation="horizontal", aspect=20)
        cb.ax.tick_params(labelsize=20)

        # Put ticks on top of the bar
        ax_cbar.xaxis.set_ticks_position("top")

        plt.close()

        fig, ax = plt.subplots(figsize=(10, 10))
        sc = ax.scatter(x=full_latent[:, 1], y=full_latent[:, 2], c=alpha_label, alpha=0.5, cmap="viridis")

    def sample_one_reconstruction(self, x):
        with torch.no_grad():
            z_mean, z_logvar = self.model.forward(x)
            z = reparametrize(z_mean, z_logvar)
            # z = z_mean
            decoder_mean = self.model.decoder(z)
        return decoder_mean[:, 0, :]

    def reconstruction_loss_plot(self, x, mu_x):
        batch_size = x.size(0)
        r_loss = 0.5 * F.mse_loss(x, mu_x, reduction="sum").div(batch_size)
        return r_loss

    # Plot the reconstruction for a random batch taken from the loader
    def plot_reconstruction(self, loader, title):
        x_p, _ = next(iter(loader))
        _, z_logvar = self.model.forward(x_p)
        decoder_mean = self.sample_one_reconstruction(x_p)

        for i in range(x_p.shape[0]):
            # Plot the Original vs reconstruction
            x_or = x_p[i, 0, :].detach().numpy()
            x_pos = np.arange(len(x_or))
            x_rec = decoder_mean[i, :].detach().numpy()

            plt.figure(f"train_{i}", figsize=(20, 10))
            _, stemlines1, _ = plt.stem(x_pos, x_or, "tab:blue", markerfmt=" ", label="Original")
            plt.setp(stemlines1, "linewidth", 3)
            _, stemlines2, _ = plt.stem(x_pos + 0.3, x_rec, "tab:orange", markerfmt=" ", label="Reconstruction")
            plt.setp(stemlines2, "linewidth", 3)
            plt.title(f"reconstruction for a {title} sample{i}_std_{z_logvar[i].exp().div(2)}")
            plt.legend()
            plt.close()

    # Make the latent traversal for factor{f_index} with values ranging from z_min to z_max
    def latent_traversal(self, f_index, loader, z_min, z_max):
        # inter =  10
        inter = (z_max - z_min) / 100
        x, _ = next(iter(loader))
        interpolation = torch.arange(z_min, z_max, inter)

        with torch.no_grad():
            z_mean, z_logvar = self.model.forward(x)
            z_mean = z_mean[0]
            z_logvar = z_logvar[0]
            z_ori = reparametrize(z_mean, z_logvar)

        time = range(len(interpolation))
        # create all images
        for t in time:
            self.create_frame(z_ori, f_index, t, interpolation)
        # Save all images in frames
        frames = []
        for t in time:
            image = imageio.v2.imread(f"plots/latent_traversal/images/latent_factor{f_index}_step_t{t}.png")
            frames.append(image)
        imageio.mimsave(
            f"plots/latent_traversal/gifs/latent_traversal_f{f_index}.gif",  # output gif
            frames,  # array of input frames
            fps=5,
        )  # optional: frames per second

    # Make {n_points} different latent traversal between pol_limits1 and pol_limits 2
    def latent_traversal_legs(self, mask_to_name, pol_limits1, pol_limits2):
        n_points = 50
        p1 = m_path.Path(pol_limits1)
        p2 = m_path.Path(pol_limits2)
        flag1 = p1.contains_points(self.full_latent[:, 1:])
        flag2 = p2.contains_points(self.full_latent[:, 1:])

        n_steps = 15
        t = np.abs(np.max(self.full_latent[flag2, 1]) - np.min(self.full_latent[flag1, 1])) / n_steps

        # Plot the ellipses of latent space
        _, ax = plt.subplots(figsize=(10, 10))
        plt.xlim((-4.5, 4.5))
        plt.ylim((-4.5, 4.5))

        c_dict = self.color_dict()
        c_dict = {0: "blue"}

        poly_colors = ["#a30000", "black"]

        self.scatter_with_covar(ax, self.full_latent[:, 1:], self.vars, self.label, c_dict, mask_to_name)
        poly1 = Polygon(pol_limits1, alpha=0.5, ec="gray", fc=poly_colors[0], visible=True)
        ax.add_patch(poly1)
        poly2 = Polygon(pol_limits2, alpha=0.5, ec="gray", fc=poly_colors[1], visible=True)
        ax.add_patch(poly2)

        random_start = np.random.choice(np.sum(flag1), size=n_points, replace=False)
        z_start = self.full_latent[flag1, 1:][random_start]

        p_to_steps = {}
        all_points = []
        for i in range(n_points):
            ax.plot(z_start[i, 0], z_start[i, 1], ".")
            z_curr = z_start[i, :]
            j = 0
            latent_val = [z_curr]
            while not p2.contains_point(z_curr) and j < n_steps:
                plt.quiver(z_curr[0], z_curr[1], t, 0, scale_units="xy", angles="xy", scale=1, alpha=0.8)
                z_curr = z_curr + np.array([t, 0])
                latent_val.append(z_curr)
                j += 1
            all_points.append(latent_val)
            p_to_steps[i] = len(latent_val)
        return all_points

    # all_points = list of latent traversal
    def get_reconstruction(self, all_points, d):
        all_reconstructed = []
        for latent_val in all_points:
            one_traversal = []
            one_traversal = np.zeros((len(latent_val), d))
            for t, l in enumerate(latent_val):
                z = torch.Tensor(l)
                with torch.no_grad():
                    decoder_mean = self.model.decoder(z)[0, 0, :]
                    intensity = decoder_mean.detach().numpy()
                one_traversal[t, :] = intensity
            all_reconstructed.append(one_traversal)

        return all_reconstructed

    def variance_latent_traversal(self, all_points, mzs, d):
        combined_p = self.get_reconstruction(all_points, d)

        ## Subset of x_ticks that are going to be ploted
        subset = np.arange(0, combined_p[0].shape[1], step=4)
        sign_traversal = np.zeros((len(combined_p), combined_p[0].shape[1]))
        for i, p_steps in enumerate(combined_p):
            ## If this difference is positive it means the signal was increasing
            sign_traversal[i, :] = (p_steps[-1, :] - p_steps[0, :] >= 0) * 2 - 1  ## Casting values to the range [-1,1]

        ## For interpretation note that some traversal increase the value and some decrease it
        sign_traversal = np.sum(sign_traversal, axis=0) >= 0

        ## Plot of the stacked traversal
        data = {}
        for i, p_steps in enumerate(combined_p):
            data[f"var_{i}"] = np.var(p_steps, axis=0)

        sum_var = np.sum(np.array(list(data.values())), axis=0)
        col_data = pd.DataFrame({"mzs": mzs, "variance": sum_var, "increase": sign_traversal})
        col_data = col_data.sort_values(by="variance", ascending=False)
        ## Plot of the variance in order :
        col_data = col_data.sort_values(by="variance", ascending=False)

        f, ax = plt.subplots(figsize=(15, 5))

        subset = np.arange(0, 212, step=4)
        ax.set_xticks(subset)
        ticks_labels_mzs = np.array(col_data["mzs"])
        ticks_labels = np.array([f"{ticks_labels_mzs[i]:.3f}" for i in subset])
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
        ticks_labels_in = np.array([f"{ticks_labels_mzs[i]:.3f}" for i in subset])
        axins.set_xticks(subset)
        axins.set_xticklabels(ticks_labels_in, rotation=45, ha="right")

        # Plot the rest
        col_data_sub = col_data[cutx:]["increase"]
        for l in np.unique(col_data_sub):
            ind = col_data_sub == l
            X = X2[ind]
            Y = Y2[ind]
            ax.scatter(X, Y, c=color_bars[l], label=label_bars[l])

        mark_inset(ax, axins, loc1=1, loc2=3)
        ax.legend(prop={"size": 15})

    ## index_missing = Boolean to indicate if latent_val contains the index in first column
    def latent_traversal_app_gif(self, latent_val, label, f_index, n, index_missing=False):
        col_dict = self.color_dict()
        label_unique = np.unique(label)
        colors = [col_dict[l] for l in label]
        for t in range(latent_val.shape[0]):
            z = torch.Tensor(latent_val[t,]) if index_missing else torch.Tensor(latent_val[t, 1:])
            self.create_frame_app(z, t, n, f_index, colors[t], label_unique, col_dict)
        frames = []
        for t in range(latent_val.shape[0]):
            image = imageio.v2.imread(
                f"plots/latent_traversal/images/approximate/latent{n}_factor{f_index}_step_t{t}.png"
            )
            frames.append(image)
        imageio.mimsave(
            f"plots/latent_traversal/gifs/approximate/latent{n}_traversal_f{f_index}.gif",  # output gif
            frames,  # array of input frames
            fps=5,
        )  # optional: frames per second

    def latent_traversal_heatmap(self, f_index, loader, z_min, z_max):
        inter = (z_max - z_min) / 100
        x, _ = next(iter(loader))
        interpolation = torch.arange(z_min, z_max, inter)

        with torch.no_grad():
            z_mean, z_logvar = self.model.forward(x)
            z_mean = z_mean[0]
            z_logvar = z_logvar[0]
            z = reparametrize(z_mean, z_logvar)
        time = range(len(interpolation))
        data = np.zeros((x.shape[2], len(time)))
        for i, t in enumerate(time):
            z[f_index] = interpolation[t]
            decoder_mean = self.model.decoder(z)[0, 0, :]
            data[:, i] = decoder_mean.detach().numpy()
        plt.figure()
        v_min = np.min(data)
        v_max = np.max(data)
        c = plt.pcolor(data, edgecolors="k", linewidths=1e-1, cmap="RdBu", vmin=v_min, vmax=v_max)
        plt.colorbar(c)
        plt.xlabel("Time steps")
        plt.ylabel("mz response")
        plt.title(f"Latent traversal for increasing values of factor_{f_index} ")

    def color_dict(self, n_col=6):
        new_colors = ["#1f77b4", "darkorange", "green", "firebrick", "black", "darkmagenta"]
        ## For SYNTHETIC DATA
        cmap_spa = "cmr.pride"
        new_colors = cmr.take_cmap_colors(cmap_spa, n_col, return_fmt="hex")
        new_colors[-1] = [0.0, 0.5, 1.0, 1.0]  # Set the last color to blue
        c_dict = dict(zip(range(len(new_colors)), new_colors))
        return c_dict

    ## For now only works with 2 dimension
    def next_z_traversal(self, fa, z_curr, idx_curr, z_start, t, full_latent, mode="forward"):
        threshold = 0.1  # must be multidimensional when z_dim > 2
        index = np.abs(full_latent[:, 2 - fa] - z_start[1 - fa]) <= threshold  # Eliminate z_2 that are two far
        z_list = full_latent[index, :]
        if len(z_list) > 0:
            if mode == "forward":
                step_array = np.where(
                    z_list[:, 1 + fa] - z_curr[fa] + t >= 0, z_list[:, 1 + fa], np.inf
                )  # z_next >= z_c + t
                idx_closest = step_array.argmin()  # index of closest z_1
            else:
                step_array = np.where(
                    -z_list[:, 1 + fa] + z_curr[fa] + t >= 0, z_list[:, 1 + fa], -np.inf
                )  # z_next <= z_c - t
                idx_closest = step_array.argmax()  # index of closest z_1
            if not np.isinf(step_array[idx_closest]):
                z_next = z_list[idx_closest, 1:]
                idx_next = z_list[idx_closest, 0]
                return z_next, idx_next
        return z_curr, idx_curr

    def compute_full_trajectory(self, fa, z_start, idx_start, t, n_steps, full_latent):
        z_arr1, idx_arr1 = self.compute_trajectory(fa, z_start, idx_start, t, n_steps, full_latent, mode="forward")
        z_arr2, idx_arr2 = self.compute_trajectory(fa, z_start, idx_start, t, n_steps, full_latent, mode="backward")
        z_arr = np.concatenate((np.flip(z_arr2, axis=0), z_arr1[1:,]), axis=0)
        z_idx = np.concatenate((np.flip(idx_arr2, axis=0), idx_arr1[1:,]), axis=0)

        z_idx = [int(key) for key, _ in groupby(z_idx)]
        return z_arr, z_idx

    # z_start : point on which the trajectory starts
    # t : min space until next point on traversal
    # n_steps : number of steps on the trajectory
    # full_latent : 2d matrix containing the full latent space
    def compute_trajectory(self, fa, z_start, idx_start, t, n_steps, full_latent, mode="forward"):
        idx_arr = np.zeros(n_steps + 1)
        z_arr = np.zeros((n_steps + 1, self.model.z_dim))
        idx_arr[0] = idx_start
        z_arr[0, :] = z_start
        for j in range(1, n_steps + 1):
            z_next, idx_next = self.next_z_traversal(fa, z_arr[j - 1, :], idx_arr[j - 1], z_start, t, full_latent, mode)
            idx_arr[j] = idx_next
            z_arr[j, :] = z_next

        return z_arr, idx_arr

    def plot_recons_dis_trade_of(self):
        _, axs = plt.subplots(3, 1, figsize=(10, 10))
        betas = [1, 2, 3, 4, 8]
        colors = ["#1d6bb3", "#e8710a", "#1b9c10", "#e52592", "#9334e6", "#f9ab00", "#12b5cb", "#df159a"]
        if len(betas) > len(colors):
            betas = betas[: len(colors)]

        colors = colors[: len(betas)]
        col_beta = dict(zip(betas, colors))
        n = 4  # Number of versions

        for b in betas:
            train_loss, train_recons, train_kl, test_loss, dis_metric = compute_loss(b, n)

            x_range = range(train_loss.shape[1])
            for i in range(n):
                axs[0].plot(x_range, train_loss[i, :], color=col_beta[b], alpha=0.5)
                axs[1].plot(x_range, train_recons[i, :], color=col_beta[b], alpha=0.5)
            axs[0].plot(x_range, np.mean(train_loss, axis=0), color=col_beta[b], label=b)
            axs[0].title.set_text("Training loss")

            axs[1].plot(x_range, np.mean(train_recons, axis=0), col_beta[b], label=b)
            axs[1].title.set_text("Reconstruction error")

            mean_dis = np.mean(dis_metric, axis=0)
            sd_dis = np.std(dis_metric, axis=0)
            axs[2].plot(x_range, mean_dis, color=col_beta[b], label=b)
            axs[2].fill_between(x_range, mean_dis - sd_dis, mean_dis + sd_dis, alpha=0.3)
            axs[2].title.set_text("Disentangling metric")
            axs[2].set_xlabel("Epochs")

            for j in range(3):
                axs[j].legend().get_frame().set_linewidth(0.0)
                axs[j].legend(title=r"$\beta$", bbox_to_anchor=(1, 1), loc="upper left")

    def plot_spatial(self, indices):
        ############# For kidney data
        ## Draw image mask on one mzs in the bottom
        centroids, _, pixel_index = dat.load()  # original data
        image_shape, norm, mzs = dat.load_shape_norm_mzs()
        # index_to_pos = self.index_to_image_pos(image_shape, pixel_index)

        for i in indices:
            plt.figure(figsize=(10, 10))
            plt.imshow(dat.reshape_array(centroids[:, i] / norm, image_shape, pixel_index))
            plt.title(f"m/z {mzs[i]:.4f}", fontsize=20)
            plt.savefig(self.output_dir / f"plots/spatial/mz_{i}", bbox_inches="tight")

    def plot_polygons_get_mask(self, ax, pol_limits, colors, index_to_pos, image_shape):
        masks = np.zeros(image_shape)
        for i, pol_limit in enumerate(pol_limits):
            
            ## Polygones on top images
            poly1 = Polygon(pol_limit, alpha=0.5, ec="gray", fc=colors[i], visible=True)
            p = m_path.Path(pol_limit)
            ax.add_patch(poly1)
            flag = p.contains_points(self.full_latent[:, 1:])
            index_mask = self.full_latent[:, 0][flag]
            
            ###### Selection for traversal
            # min_d = np.argmin(self.full_latent[flag,2])
            # max_d = np.argmax(self.full_latent[flag,2])
            pos = np.zeros((len(index_mask), 2))
            for j, ind in enumerate(index_mask):
                pos[j, :] = index_to_pos[ind]

            ## Plot scatter image bottom right
            for j, ind in enumerate(index_mask):
                masks[int(pos[j, 0]), int(pos[j, 1])] = i + 1
        return masks

    def plot_kernel_density_estimation(self, full_latent, vars, label, mask_to_name, title=""):
        _, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        # Create grid to evaluate the kernel on
        X, Y = np.mgrid[-4:4:200j, -4:4:200j]
        grid = np.vstack([X.ravel(), Y.ravel()])

        # Define Kernel Scott
        kernel = stats.gaussian_kde(full_latent[:, 1:].T, bw_method="scott")
        kernel.set_bandwidth(bw_method="scott")
        factor = kernel.factor
        factors = np.zeros((2, 2))
        factors[0, 1] = factor
        factors[1, 0] = factor / 1.5
        factors[1, 1] = factor / 3

        for i in range(2):
            for j in range(2):
                if i == 0 and j == 0:
                    prop = 0.05
                    prop = 0.2
                    sub_index = np.random.choice(full_latent.shape[0], int(full_latent.shape[0] * prop), replace=False)
                    self.scatter_with_covar(
                        ax[0, 0],
                        full_latent[sub_index, 1:],
                        vars[sub_index],
                        label[sub_index],
                        self.color_dict(),
                        mask_to_name,
                    )
                    ax[0, 0].set_xlim(-4, 4)
                    ax[0, 0].set_ylim(-4, 4)
                else:
                    self.kernel_density(ax[i, j], kernel, factors[i, j], grid, X.shape)

    def kernel_density(self, ax, fig, kernel, factor, grid, shape):
        kernel.set_bandwidth(bw_method=factor)
        Z = np.reshape(kernel(grid).T, shape)
        im = ax.imshow(np.rot90(Z), cmap=plt.cm.jet, extent=[-4, 4, -4, 4])

        # add color bar above picture
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size="5%", pad=0.5)
        fig.add_axes(cax)
        colorbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        # Set the colorbar ticks and label size
        colorbar.ax.tick_params(labelsize=20)

    def do_plot(self, ax, Z, transform, cmap_col):
        im = ax.imshow(
            Z,
            interpolation="none",
            origin="lower",
            cmap=cmap_col,
            # extent=[-2, 4, -3, 2],
            clip_on=True,
            alpha=0,
        )

        trans_data = transform + ax.transData
        im.set_transform(trans_data)

        # display intended extent of the image
        x1, x2, y1, y2 = im.get_extent()
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--", transform=trans_data, linewidth=5)

    def reg(self, r, width, height):
        return [r[0], r[0] + width, r[1], r[1] + height]
