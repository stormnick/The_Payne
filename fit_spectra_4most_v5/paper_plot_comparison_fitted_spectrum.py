from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 17.07.25

if __name__ == '__main__':
    # Load spectra
    wavelength_fitted, flux_fitted = np.loadtxt(
        "fitted_spectrum_HD140283_UVES.txt", usecols=(0, 1), unpack=True, dtype=float
    )
    wavelength_fitted, flux_fitted = np.loadtxt(
        "fitted_spectrum_HD140283_UVES_4most.txt", usecols=(0, 1), unpack=True, dtype=float
    )
    wavelength_obs, flux_obs = np.loadtxt(
        "/Users/storm/PhD_2025/02.22 Payne/real_spectra_to_fit/converted/UVES_HD140283_v2.txt",
        usecols=(0, 1), unpack=True, dtype=float
    )
    wavelength_obs, flux_obs = np.loadtxt(
        "/Users/storm/PhD_2022-2025/Spectra/some_lowfeh_benchmark/norm_20k_degraded/HD140283_UVES_4most.txt",
        usecols=(0, 1), unpack=True, dtype=float
    )

    # reinterpolate the observed wavelength grid to dlam = 0.01
    dlam = 0.05
    wavelength_obs_interp = np.arange(wavelength_obs.min(), wavelength_obs.max(), dlam)
    interp_func = np.interp(wavelength_obs_interp, wavelength_obs, flux_obs)
    flux_obs = interp_func
    wavelength_obs = wavelength_obs_interp

    extra_dlam = 5

    # Define the windows to display in the main (broken-axis) plot
    windows = [(3926 - extra_dlam, 4355 + extra_dlam), (5160 - extra_dlam, 5730 + extra_dlam), (6100 - extra_dlam, 6790 + extra_dlam)]

    # Define zoom regions (e.g. Li line at ~6707 Å ±0.5 Å)
    zoom_regions = [
        (4290.01, 4329.99),
        (5166.51, 5184.49),
        #(5166.71, 5168.00),
        (6707.21, 6708.349),  # Li line
        # Add more regions as needed
    ]

    telluric_region = (6275, 6557)

    # Create a mosaic layout: 3 main panels on top, matching zoom panels below
    mosaic = [
        ['m1', 'm2', 'm3'],
        ['z1', 'z2', 'z3'],
    ]
    titles = ['CH G-band', 'Mg I b triplet', 'Li I doublet']
    fig, axs = plt.subplot_mosaic(
        mosaic,
        figsize=(16, 8),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.26, 'wspace': 0.09}
    )

    fig.suptitle('HD 140283 4MOST-HR Payne fit', fontsize=16, y=0.92)

    # -- Plot main broken-axis panels --
    for i, ax_key in enumerate(['m1', 'm2', 'm3']):
        ax = axs[ax_key]
        w0, w1 = windows[i]
        mask_obs = (wavelength_obs >= w0) & (wavelength_obs <= w1)
        mask_fit = (wavelength_fitted >= w0) & (wavelength_fitted <= w1)
        #ax.plot(wavelength_obs[mask_obs], flux_obs[mask_obs], color='black', lw=0.6)
        ax.scatter(wavelength_obs[mask_obs], flux_obs[mask_obs], color='black', s=0.7, rasterized=True)
        ax.plot(wavelength_fitted[mask_fit], flux_fitted[mask_fit], color='red', lw=0.6)
        ax.set_xlim(w0, w1)
        ax.set_ylim(0, flux_obs[mask_obs].max() + 0.01)
        ax.tick_params(axis='both', which='major', labelsize=11)
        if i != 0:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
        if i != len(windows) - 1:
            ax.spines['right'].set_visible(False)
        if i == 0:
            ax.set_ylabel('Normalised Flux')


    axs['m2'].set_xlabel('Wavelength [Å]')

    # Draw diagonal lines to indicate breaks
    d = 0.015 * (axs['m1'].get_ylim()[1] - axs['m1'].get_ylim()[0])
    for i in [0, 1]:
        ax_left = axs[f'm{i+1}']
        ax_right = axs[f'm{i+2}']
        kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False)
        ax_left.plot((1, 1 + 0.02), (-d, +d), **kwargs)
        kwargs = dict(transform=ax_right.transAxes, color='k', clip_on=False)
        ax_right.plot((-0.02, 0), (-d, +d), **kwargs)

    # -- Plot zoom panels below and draw connectors --
    for i, ax_key in enumerate(['z1', 'z2', 'z3']):
        axz = axs[ax_key]
        if i < len(zoom_regions):
            zr0, zr1 = zoom_regions[i]
            # Mask zoom region
            mask_obs_z = (wavelength_obs >= zr0) & (wavelength_obs <= zr1)
            mask_fit_z = (wavelength_fitted >= zr0) & (wavelength_fitted <= zr1)
            #axz.plot(wavelength_obs[mask_obs_z], flux_obs[mask_obs_z], color='black', lw=0.6)
            axz.scatter(wavelength_obs[mask_obs_z], flux_obs[mask_obs_z], color='black', s=2, rasterized=True)
            axz.plot(wavelength_fitted[mask_fit_z], flux_fitted[mask_fit_z], color='red', lw=0.6)
            axz.set_xlim(zr0, zr1)
            if i == 0:
                axz.set_ylabel('Normalised Flux')
            # Draw connecting lines between main and zoom
            main_ax = axs[f'm{i+1}']
            # Fractional x positions in main axis coords
            x0 = (zr0 - windows[i][0]) / (windows[i][1] - windows[i][0])
            x1 = (zr1 - windows[i][0]) / (windows[i][1] - windows[i][0])
            # Connector from left edge of zoom to main
            axz.annotate('', xy=(x0, 0), xycoords=main_ax.transAxes,
                         xytext=(0, 1), textcoords=axz.transAxes,
                         arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))
            # Connector from right edge of zoom to main
            axz.annotate('', xy=(x1, 0), xycoords=main_ax.transAxes,
                         xytext=(1, 1), textcoords=axz.transAxes,
                         arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

            # vertical lines at zoom edges in the top panel
            main_ax.axvline(zr0, color='gray', linestyle='--', lw=0.5)
            main_ax.axvline(zr1, color='gray', linestyle='--', lw=0.5)

            axz.set_ylim(flux_obs[mask_obs_z].min() - 0.02, flux_obs[mask_obs_z].max() + 0.02)

            # set ticks fontsize
            axz.tick_params(axis='both', which='major', labelsize=11)

            # yticks every 0.1
            yticks = np.arange(
                np.floor(axz.get_ylim()[0] * 10) / 10,
                1 + 0.05, 0.1
            )
            axz.set_yticks(yticks)

            # set title
            axz.set_title(titles[i], fontsize=12)
        else:
            axz.axis('off')

    axs['z2'].set_xlabel('Wavelength [Å]')

    # after you’ve plotted m3 (i.e. axs['m3']):
    m3 = axs['m3']

    y_base = 0.825  # vertical position of the horizontal line
    h = 0.02  # height of the little vertical ticks

    # left vertical tick
    m3.plot([telluric_region[0], telluric_region[0]], [y_base, y_base + h], color='gray', lw=1)
    # horizontal connector
    m3.plot([telluric_region[0], telluric_region[1]], [y_base, y_base], color='gray', lw=1)
    # right vertical tick
    m3.plot([telluric_region[1], telluric_region[1]], [y_base, y_base + h], color='gray', lw=1)

    # label centered under the bar
    m3.text(
        (telluric_region[0] + telluric_region[1]) / 2,  # x-position
        y_base - 0.01,  # a bit below the line
        'Telluric lines in\nobserved spectrum',  # your label
        ha='center', va='top',  # center horizontally, top aligned to that y
        fontsize=10,
        color='gray'
    )

    m3.scatter([-99], [-99], color='black', s=15, label='Observed')
    m3.plot([-99], [-99], color='red', lw=2, label='Payne Fitted')
    m3.legend(fontsize=12, loc='lower left', frameon=False)

    # for z2 i need tick seprartion of 5 IT DOESNT WORK
    axs['z2'].xaxis.set_major_locator(plt.MultipleLocator(5))

    plt.savefig("../plots/HD140283_4MOST-HR_Payne_fit.pdf", bbox_inches='tight')
    plt.show()