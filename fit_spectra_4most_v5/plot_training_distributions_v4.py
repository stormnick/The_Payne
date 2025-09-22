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
from matplotlib.colors import Normalize

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize

MIN_COUNT = 20      # bins with < MIN_COUNT will be white
MAX_COUNT = 400    # bins with > MAX_COUNT will use the max colour
NBINS = 50

if __name__ == '__main__':
    data1 = pd.read_csv("spectra_parameters_6may2025_batch0.csv")
    data2 = pd.read_csv("spectra_parameters_6may2025_batch1.csv")
    data3 = pd.read_csv("spectra_parameters_6may2025_batch2.csv")
    data_all = pd.concat([data1, data2, data3], ignore_index=True)

    print(f"data1: {len(data1)} rows, data2: {len(data2)} rows, data3: {len(data3)} rows, data_all: {len(data_all)} rows")

    data_all = data_all[(data_all['logg'] <= 5.0) & (data_all['logg'] >= 0.5)]
    data_all["A_Li"] = data_all["Li_Fe"] + data_all["feh"] + 1.05
    data_all = data_all.drop(columns=["Li_Fe"])

    print(f"Teff min: {data_all['teff'].min()}, Teff max: {data_all['teff'].max()}")

    fig, axs = plt.subplots(5, 4, figsize=(10, 11))
    axs = axs.flatten()

    # --- shared colour mapping for ALL panels (persistent colouring) ---
    cmap = plt.cm.get_cmap("plasma_r").copy()
    #cmap = plt.cm.get_cmap("rainbow").copy()
    cmap.set_bad("white")  # NaNs (masked bins) drawn as white
    norm = Normalize(vmin=MIN_COUNT, vmax=MAX_COUNT, clip=True)

    # -------- 1) teff–logg (heatmap instead of scatter) -------------------
    teff_lo, teff_hi = data_all['teff'].min(), data_all['teff'].max()
    logg_lo, logg_hi = 0.5, 5.0  # already filtered, but set explicit range
    H, xedges, yedges = np.histogram2d(
        data_all['teff'], data_all['logg'],
        bins=NBINS,
        range=[[teff_lo, teff_hi], [logg_lo, logg_hi]]
    )
    H = np.where(H < MIN_COUNT, np.nan, np.clip(H, None, MAX_COUNT))
    X, Y = np.meshgrid(xedges, yedges)
    im = axs[0].pcolormesh(X, Y, H.T, cmap=cmap, norm=norm, shading='auto', rasterized=True)
    axs[0].set_xlabel('Teff [K]', fontsize=14)
    axs[0].set_ylabel('logg', fontsize=14)
    axs[0].set_xlim(teff_lo, teff_hi)
    axs[0].set_ylim(logg_lo, logg_hi)
    axs[0].invert_yaxis()
    axs[0].invert_xaxis()

    # -------- 2) [Fe/H]–logg (heatmap instead of scatter) -----------------
    feh_lo, feh_hi = -5.1, 0.51
    H, xedges, yedges = np.histogram2d(
        data_all['feh'], data_all['logg'],
        bins=NBINS,
        range=[[feh_lo, feh_hi], [logg_lo, logg_hi]]
    )
    H = np.where(H < MIN_COUNT, np.nan, np.clip(H, None, MAX_COUNT))
    X, Y = np.meshgrid(xedges, yedges)
    im = axs[1].pcolormesh(X, Y, H.T, cmap=cmap, norm=norm, shading='auto', rasterized=True)
    axs[1].text(0.07, 0.91, "logg", transform=axs[1].transAxes, ha='left', va='top', fontsize=16, color='white')
    axs[1].set_xlim(feh_lo, feh_hi)
    axs[1].set_ylim(logg_lo, logg_hi)

    # -------- 3) [Fe/H]–vmic (heatmap instead of scatter) -----------------
    vmic_lo = data_all['vmic'].min()
    vmic_hi = data_all['vmic'].max()
    H, xedges, yedges = np.histogram2d(
        data_all['feh'], data_all['vmic'],
        bins=NBINS,
        range=[[feh_lo, feh_hi], [vmic_lo, vmic_hi]]
    )
    H = np.where(H < MIN_COUNT, np.nan, np.clip(H, None, MAX_COUNT))
    X, Y = np.meshgrid(xedges, yedges)
    im = axs[2].pcolormesh(X, Y, H.T, cmap=cmap, norm=norm, shading='auto', rasterized=True)
    axs[2].text(0.07, 0.91, "vmic [km/s]", transform=axs[2].transAxes, ha='left', va='top', fontsize=16, color='white')
    axs[2].set_xlim(feh_lo, feh_hi)
    axs[2].set_ylim(vmic_lo, vmic_hi)

    # -------- Remaining element panels (unchanged, share same cmap/norm) ---
    for i, element in enumerate(data_all.columns[6:]):
        counts, xedges, yedges = np.histogram2d(
            data_all['feh'],
            data_all[element],
            bins=NBINS,
            range=[[feh_lo, feh_hi],
                   [np.min(data_all[element]) - 0.06,
                    np.max(data_all[element]) + 0.06]]
        )
        counts = np.where(counts < MIN_COUNT, np.nan, np.clip(counts, None, MAX_COUNT))

        X, Y = np.meshgrid(xedges, yedges)
        im = axs[i + 3].pcolormesh(X, Y, counts.T, cmap=cmap, norm=norm, shading='auto', rasterized=True)

        dist_width = np.max(data_all[element]) - np.min(data_all[element])
        text = 'A(Li)' if element == "A_Li" else f'[{element.replace("_Fe", "/Fe]")}'
        axs[i + 3].text(0.07, 0.91, text, transform=axs[i + 3].transAxes,
                        ha='left', va='top', fontsize=16, color='white')

        if dist_width <= 1.8:
            axs[i + 3].yaxis.set_major_locator(MultipleLocator(0.5))
            axs[i + 3].yaxis.set_minor_locator(MultipleLocator(0.1))
        else:
            axs[i + 3].yaxis.set_major_locator(MultipleLocator(1))
            axs[i + 3].yaxis.set_minor_locator(MultipleLocator(0.2))

        axs[i + 3].set_xlim(feh_lo, feh_hi)
        axs[i + 3].set_ylim(np.min(data_all[element]) - 0.06,
                            np.max(data_all[element]) + 0.06)

    # --- single colourbar on the right for all subplots -------------------
    # make a new axis for the colourbar (left, bottom, width, height)
    cax = fig.add_axes([0.91, 0.1, 0.03, 0.78])  # tweak 0.92 → move right/left
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Count per bin", fontsize=14)

    # tick label font size
    cbar.ax.tick_params(labelsize=12)

    # --- layout/ticks tweaks (your existing logic) ------------------------
    axs2d = axs.reshape(4, 5)
    fig.subplots_adjust(hspace=0.00, wspace=0.25)

    ax00 = axs2d[0, 0]
    ax00.xaxis.set_label_position('top')
    ax00.xaxis.tick_top()
    ax00.tick_params(axis='x', labelbottom=False)

    for r in range(4):
        for c in range(5):
            ax = axs2d[r, c]
            if r == 0 and c == 0:
                continue
            if r < 2:
                ax.tick_params(axis='x', labelbottom=False)
            if r < 3:
                ax.set_xlabel('')

    fig.supxlabel('[Fe/H]', y=0.05, fontsize=18)
    fig.supylabel('[X/Fe]', x=0.075, fontsize=18)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.savefig("../plots/training_distributions_6may2025_batch0-1.pdf", bbox_inches='tight', dpi=300)
    plt.show()