from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_scatter_density import ScatterDensityAxes
import scipy.ndimage as ndi
from fastkde import fastKDE
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

matplotlib.use("MacOSX")
plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 08.08.25

def return_density_plot2(x, y, axarr_to_plot):
    indices_not_nan = ~np.isnan(y)

    xy_point_density = np.vstack([x[indices_not_nan], y[indices_not_nan]])
    z_point_density = gaussian_kde(xy_point_density)(xy_point_density)

    idx_sort = z_point_density.argsort()
    x_plot, y_plot, z_plot = x[indices_not_nan][idx_sort], y[indices_not_nan][idx_sort], z_point_density[idx_sort]

    def truncate_colormap(cmap, minval=0.0, maxval=0.8, n=256):
        """Return a truncated copy of *cmap* from *minval* to *maxval*."""
        new_colors = cmap(np.linspace(minval, maxval, n))
        return mcolors.LinearSegmentedColormap.from_list(
            f"{cmap.name}_trunc_{minval}_{maxval}", new_colors)

    plasma80 = truncate_colormap(cm.get_cmap("plasma_r"), 0.2, 1)

    # --- plotting ----------------------------------------------------------------
    density = axarr_to_plot.scatter(
        x_plot, y_plot,
        c=z_plot,
        s=20,
        cmap=plasma80,  # ⇦ use the truncated map
        alpha=1,
        zorder=-1
    )

def return_density_plot(fig, x, y, axarr_to_plot):

    cmap = cm.get_cmap('plasma_r').copy()  # make a *mutable* copy
    cmap.set_under('white', alpha=1)  # or alpha=0 for transparency


    x, y = np.asarray(x), np.asarray(y)
    pdf = fastKDE.pdf(x, y)  # seconds, not minutes
    norm = mcolors.Normalize(vmin=1e-3, vmax=1, clip=False)
    pdf.plot(ax=axarr_to_plot, add_colorbar=False, cmap=cmap, shading='auto', rasterized=True, norm=norm)

    return
    indices_not_nan = ~np.isnan(y)
    x, y = x[indices_not_nan], y[indices_not_nan]

    axarr_to_plot = ScatterDensityAxes(fig, rect=axarr_to_plot.get_position())
    fig.add_axes(axarr_to_plot)

    axarr_to_plot.scatter_density(x, y, cmap=cmap, rasterized=True)

    return
    x, y = np.asarray(x), np.asarray(y)
    pdf = fastKDE.pdf(x, y)  # seconds, not minutes
    print(type(pdf))
    pdf.plot(ax=axarr_to_plot, add_colorbar=False, cmap='plasma_r', shading='auto',
             vmin=0, rasterized=True)
    #axarr_to_plot.pcolormesh(xs, ys, dens.T, cmap='plasma_r', shading='auto')

    return
    H, xedges, yedges = np.histogram2d(x, y, bins=300)
    # optional: Gaussian blur to smooth the sharp bin edges
    H = ndi.gaussian_filter(H, sigma=1.2)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    axarr_to_plot.imshow(H.T, extent=extent, origin='lower', aspect='auto', cmap='plasma_r',
                         norm=mcolors.LogNorm(vmin=1, vmax=H.max()), interpolation='bilinear')

if __name__ == '__main__':
    data1 = pd.read_csv("spectra_parameters_6may2025_batch0.csv")
    data2 = pd.read_csv("spectra_parameters_6may2025_batch1.csv")
    data3 = pd.read_csv("spectra_parameters_6may2025_batch2.csv")
    data_all = pd.concat([data1, data2, data3], ignore_index=True)

    # print how many rows
    print(f"data1: {len(data1)} rows, data2: {len(data2)} rows, data3: {len(data3)} rows, data_all: {len(data_all)} rows")

    # only take data_all with logg <= 5.0 and >= 0.5
    data_all = data_all[(data_all['logg'] <= 5.0) & (data_all['logg'] >= 0.5)]
    data_all["A_Li"] = data_all["Li_Fe"] + data_all["feh"] + 1.05
    # drop Li_Fe
    data_all = data_all.drop(columns=["Li_Fe"])

    # print min teff and max teff
    print(f"Teff min: {data_all['teff'].min()}, Teff max: {data_all['teff'].max()}")

    # plot distributions of parameters
    # teff-logg, feh-vmic, all-elements-feh
    fig, axs = plt.subplots(4, 5, figsize=(12, 9))

    alpha = 0.16

    axs = axs.flatten()
    axs[0].scatter(data_all['teff'], data_all['logg'], s=0.01, alpha=alpha, c='k', rasterized=True)
    axs[0].set_xlabel('Teff [K]', fontsize=14)
    axs[0].set_ylabel('logg', fontsize=14)
    axs[0].invert_yaxis()
    axs[0].invert_xaxis()

    axs[1].scatter(data_all['feh'], data_all['logg'], s=0.01, alpha=alpha, c='k', rasterized=True)
    #axs[1].set_xlabel('[Fe/H]')
    #axs[1].set_ylabel('logg')
    axs[1].text(0.07, 0.91, "logg", transform=axs[1].transAxes, ha='left', va='top', fontsize=16, color='white')

    axs[2].scatter(data_all['feh'], data_all['vmic'], s=0.01, alpha=alpha, c='k', rasterized=True)
    #axs[2].set_xlabel('[Fe/H]')
    axs[2].text(0.07, 0.91, "vmic [km/s]", transform=axs[2].transAxes, ha='left', va='top', fontsize=16, color='white')

    elements = data_all.columns[6:]
    # order them

    for i, element in enumerate(data_all.columns[6:]):
        axs[i + 3].scatter(data_all['feh'], data_all[element], s=0.01, alpha=alpha, c='k', rasterized=True)

        dist_width = np.max(data_all[element]) - np.min(data_all[element])

        # put text top left with the element
        if element == "A_Li":
            text = f'A(Li)'
        else:
            text =  f'[{element.replace("_Fe", "/Fe]")}'
        axs[i + 3].text(0.07, 0.91, text, transform=axs[i + 3].transAxes, ha='left', va='top', fontsize=16, color='white')

        if dist_width <= 1.8:
            axs[i + 3].yaxis.set_major_locator(MultipleLocator(0.5))  # dy = 5
            axs[i + 3].yaxis.set_minor_locator(MultipleLocator(0.1))
        else:
            axs[i + 3].yaxis.set_major_locator(MultipleLocator(1))  # dy = 5
            axs[i + 3].yaxis.set_minor_locator(MultipleLocator(0.2))
        axs[i + 3].set_xlim(-5.1, 0.51)
        axs[i + 3].set_ylim(np.min(data_all[element]) - 0.06, np.max(data_all[element]) + 0.06)

    # --- turn the flat list back into a 4×5 grid --------------------------
    axs2d = axs.reshape(4, 5)  # easier to reason in rows/cols

    # ---------------------------------------------------------------------
    # 1. kill the vertical gaps and keep a little horizontal breathing room
    fig.subplots_adjust(hspace=0.00,  # no space between rows
                        wspace=0.25)  # tweak horizontally to taste

    # 2. first subplot: ticks + label at the *top*, none at the bottom
    ax00 = axs2d[0, 0]
    ax00.xaxis.set_label_position('top')
    ax00.xaxis.tick_top()
    ax00.tick_params(axis='x', labelbottom=False)  # hide the bottom ticks

    # 3. X-tick labels logic for every other panel
    for r in range(4):  # row index
        for c in range(5):  # col index
            ax = axs2d[r, c]

            # skip the special case (already handled)
            if r == 0 and c == 0:
                continue

            # keep tick *numbers* only on the two bottom rows (rows 2 & 3)
            if r < 2:  # rows 0 and 1
                ax.tick_params(axis='x', labelbottom=False)

            # drop x-axis labels everywhere except the bottom row
            if r < 3:  # rows 0,1,2
                ax.set_xlabel('')  # no per-subplot xlabel

    # If you want one *global* x-label centred under the whole grid:
    fig.supxlabel('[Fe/H]', y=0.05, fontsize=18)  # adjust text & y-offset as needed

    # set fontsize for all axes
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=11)

    plt.savefig("../plots/training_distributions_6may2025_batch0-1.pdf", bbox_inches='tight', dpi=300)
    plt.show()