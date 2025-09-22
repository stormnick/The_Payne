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
from matplotlib.colors import LogNorm

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


#def return_density_plot3(fig, x, y, axarr_to_plot):


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
    fig, axs = plt.subplots(5, 4, figsize=(10, 11))
    axs = axs.flatten()

    # --- density settings ---
    nx = 200  # x bins
    ny = 200  # y bins
    cmap = plt.cm.get_cmap('plasma').copy()
    cmap.set_under('white')
    cmap.set_bad('white')

    # We'll compute all histograms first to get a shared vmax
    Hs = [None] * len(axs)  # to store hist arrays (transposed for pcolormesh)
    xbins_list = [None] * len(axs)  # per-panel x-edges
    ybins_list = [None] * len(axs)  # per-panel y-edges
    max_count = 1  # global maximum bin count

    # --- panel 0: Teff vs logg (density) ---
    x0 = data_all['teff'].to_numpy(np.float32)
    y0 = data_all['logg'].to_numpy(np.float32)
    m0 = np.isfinite(x0) & np.isfinite(y0)
    xb0 = np.linspace(np.nanmin(x0[m0]), np.nanmax(x0[m0]), nx + 1)
    yb0 = np.linspace(np.nanmin(y0[m0]), np.nanmax(y0[m0]), ny + 1)
    H0, _, _ = np.histogram2d(x0[m0], y0[m0], bins=[xb0, yb0])
    Hs[0] = H0.T
    xbins_list[0] = xb0
    ybins_list[0] = yb0
    max_count = max(max_count, int(H0.max()))

    # --- panels 1 & 2 share x='feh' ---
    xfeh = data_all['feh'].to_numpy(np.float32)
    xfeh_m = np.isfinite(xfeh)
    # Use your plotting range for [Fe/H] to match set_xlim later
    xb_feh = np.linspace(-5, 0.5, nx + 1)

    # panel 1: [Fe/H] vs logg
    y1 = data_all['logg'].to_numpy(np.float32)
    m1 = xfeh_m & np.isfinite(y1)
    yb1 = np.linspace(np.nanmin(y1[m1]), np.nanmax(y1[m1]), ny + 1)
    H1, _, _ = np.histogram2d(xfeh[m1], y1[m1], bins=[xb_feh, yb1])
    Hs[1] = H1.T
    xbins_list[1] = xb_feh
    ybins_list[1] = yb1
    max_count = max(max_count, int(H1.max()))

    # panel 2: [Fe/H] vs vmic
    y2 = data_all['vmic'].to_numpy(np.float32)
    m2 = xfeh_m & np.isfinite(y2)
    yb2 = np.linspace(np.nanmin(y2[m2]), np.nanmax(y2[m2]), ny + 1)
    H2, _, _ = np.histogram2d(xfeh[m2], y2[m2], bins=[xb_feh, yb2])
    Hs[2] = H2.T
    xbins_list[2] = xb_feh
    ybins_list[2] = yb2
    max_count = max(max_count, int(H2.max()))

    # --- element panels: [Fe/H] vs each abundance ---
    elements = data_all.columns[6:]  # adjust ordering if you like

    for i, element in enumerate(elements):
        idx = i + 3
        if idx >= len(axs):
            break  # safety if more elements than panels

        y = data_all[element].to_numpy(np.float32)
        m = xfeh_m & np.isfinite(y)
        if not np.any(m):
            # no data -> make a dummy empty panel
            yb = np.array([0, 1], dtype=np.float32)
            H = np.zeros((nx, 1), dtype=np.float32)
        else:
            # y-bins from min..max (change to percentiles if outliers dominate)
            yb = np.linspace(np.nanmin(y[m]), np.nanmax(y[m]), ny + 1)
            H, _, _ = np.histogram2d(xfeh[m], y[m], bins=[xb_feh, yb])

        Hs[idx] = H.T
        xbins_list[idx] = xb_feh
        ybins_list[idx] = yb
        max_count = max(max_count, int(H.max()))

    # --- shared normalization across ALL panels ---
    norm = LogNorm(vmin=1, vmax=max_count)

    # --- draw panels ---
    mappable = None

    # panel 0: Teff–logg
    H = np.ma.masked_where(Hs[0] == 0, Hs[0])
    mappable = axs[0].pcolormesh(xbins_list[0], ybins_list[0], H, cmap=cmap, norm=norm, shading='auto', rasterized=True)
    axs[0].set_xlabel('Teff [K]', fontsize=14)
    axs[0].set_ylabel('logg', fontsize=14)
    axs[0].invert_yaxis()
    axs[0].invert_xaxis()

    # panel 1: [Fe/H]–logg
    H = np.ma.masked_where(Hs[1] == 0, Hs[1])
    axs[1].pcolormesh(xbins_list[1], ybins_list[1], H, cmap=cmap, norm=norm, shading='auto', rasterized=True)
    axs[1].text(0.07, 0.91, "logg", transform=axs[1].transAxes, ha='left', va='top', fontsize=16, color='white')

    # panel 2: [Fe/H]–vmic
    H = np.ma.masked_where(Hs[2] == 0, Hs[2])
    axs[2].pcolormesh(xbins_list[2], ybins_list[2], H, cmap=cmap, norm=norm, shading='auto', rasterized=True)
    axs[2].text(0.07, 0.91, "vmic [km/s]", transform=axs[2].transAxes, ha='left', va='top', fontsize=16, color='white')

    # element panels
    for i, element in enumerate(elements):
        idx = i + 3
        if idx >= len(axs):
            break
        H = np.ma.masked_where(Hs[idx] == 0, Hs[idx])
        axs[idx].pcolormesh(xbins_list[idx], ybins_list[idx], H, cmap=cmap, norm=norm, shading='auto', rasterized=True)

        # label top-left
        if element == "A_Li":
            text = "A(Li)"
        else:
            text = f'[{element.replace("_Fe", "/Fe]")}'
        axs[idx].text(0.07, 0.91, text, transform=axs[idx].transAxes,
                      ha='left', va='top', fontsize=16, color='white')

        # tick density like before
        yvals = data_all[element].to_numpy(np.float32)
        yvals = yvals[np.isfinite(yvals)]
        if yvals.size:
            dist_width = float(np.max(yvals) - np.min(yvals))
            if dist_width <= 1.8:
                axs[idx].yaxis.set_major_locator(MultipleLocator(0.5))
                axs[idx].yaxis.set_minor_locator(MultipleLocator(0.1))
            else:
                axs[idx].yaxis.set_major_locator(MultipleLocator(1.0))
                axs[idx].yaxis.set_minor_locator(MultipleLocator(0.2))

        # match your x/y ranges
        axs[idx].set_xlim(-5., 0.5)
        if ybins_list[idx] is not None:
            axs[idx].set_ylim(ybins_list[idx][0], ybins_list[idx][-1])

    # --- single shared colorbar (log scale ticks) ---
    # --- layout & ticks like your original code ------------------------------
    axs2d = axs.reshape(5, 4)  # matches plt.subplots(5,4)

    fig.subplots_adjust(hspace=0.00, wspace=0.25)

    #cbar = fig.colorbar(mappable, ax=axs.tolist(), label='Points per bin', pad=0.01, aspect=40, fraction=0.025, location='right')
    exp_max = int(np.ceil(np.log10(max_count))) if max_count > 0 else 0
    #cbar.set_ticks([10 ** k for k in range(0, exp_max + 1)])


    # put top x-label/ticks only on [0,0]
    ax00 = axs2d[0, 0]
    ax00.xaxis.set_label_position('top')
    ax00.xaxis.tick_top()
    ax00.tick_params(axis='x', labelbottom=False)

    # X-tick labels logic for every other panel
    for r in range(5):
        for c in range(4):
            ax = axs2d[r, c]
            if r == 0 and c == 0:
                continue
            if r < 3:  # rows 0,1,2 (keep numbers only on bottom two rows)
                ax.tick_params(axis='x', labelbottom=False)
            if r < 4:  # drop per-subplot xlabel except bottom row
                ax.set_xlabel('')

    # one global x-label under the whole grid
    fig.supxlabel('[Fe/H]', y=0.02, fontsize=18)

    # set fontsize for all axes
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    plt.savefig("../plots/training_distributions_6may2025_batch0-1.pdf", bbox_inches='tight', dpi=300)
    plt.show()