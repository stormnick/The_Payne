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
        "fitted_spectrum_.txt", usecols=(0, 1), unpack=True, dtype=float
    )
    wavelength_obs, flux_obs = np.loadtxt(
        "/Users/storm/PhD_2022-2025/Spectra/some_lowfeh_benchmark/norm_20k_degraded/hd122563_UVES.txt",
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
# and 5242.49

    # Define zoom regions (e.g. Li line at ~6707 Å ±0.5 Å)
    zoom_regions = [
        (4290.01, 4329.99),
        #(5166.51, 5184.49),
        (5166.51, 5174.5),
        #(5166.71, 5168.00),
        (5527, 5529),  # Li line
        #(4121.32 - 0.7, 4121.32 + 0.7),  # Co I
        (5710, 5712.5),  # Co I
        #(5264.2 - 0.5, 5264.2 + 0.5),  #
        (5590.114 - 0.5, 5590.114 + 0.5),  #
        # Add more regions as needed
        (6555, 6570),
    ]

    lines_to_highlight = {"z1": {"CH": [4291.04091, 4292.047, 4293.102, 4295.149, 4296.946, 4296.277, 4303.87, 4306.775,
                                  4310.125, 4311.484, 4312.196, 4313.62, 4323.218, 4323.852]},
                          "z2": {"Mg I": [5167.322, 5172.684], "Fe I": [5167.49, 5171.596], "Fe II": [5169.]},
                          "z3": {"6Li": [6707.921, 6708.072], "7Li": [6707.764, 6707.915]},
                          "z5": {"Ca I": [5590.114]}, "z4": {"Co I": [4121.32]}}


    telluric_region = (6275, 6557)

    # Create a mosaic layout: 3 main panels on top, matching zoom panels below
    mosaic = [
        ['z1', 'z2', 'z3'],
        ['m1', 'm2', 'm3'],
        ['z4', 'z5', 'z6'],
    ]
    titles = ['CH G-band', 'Mg I b triplet', 'Li I doublet', 'Co I', 'Ca I', 'H-alpha']
    fig, axs = plt.subplot_mosaic(
        mosaic,
        figsize=(16, 8),
        gridspec_kw={'height_ratios': [3, 1, 3], 'hspace': 0.26, 'wspace': 0.12}
    )

    fig.suptitle(r'HD 122563: T$_{\rm eff}$ = 4747 K, logg = 0.89 dex, [Fe/H] = -2.60 dex', fontsize=16, y=0.95)

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


    #axs['m2'].set_xlabel('Wavelength [Å]')

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
    for i, ax_key in enumerate(['z1', 'z2', 'z3', 'z4', 'z5', 'z6']):
        axz = axs[ax_key]
        if i < len(zoom_regions):
            zr0, zr1 = zoom_regions[i]
            # Mask zoom region
            mask_obs_z = (wavelength_obs >= zr0 - 0.5) & (wavelength_obs <= zr1 + 0.5)
            mask_fit_z = (wavelength_fitted >= zr0 - 0.5) & (wavelength_fitted <= zr1 + 0.5)
            #axz.plot(wavelength_obs[mask_obs_z], flux_obs[mask_obs_z], color='black', lw=0.6)
            axz.scatter(wavelength_obs[mask_obs_z], flux_obs[mask_obs_z], color='black', s=2, rasterized=True)
            axz.plot(wavelength_fitted[mask_fit_z], flux_fitted[mask_fit_z], color='red', lw=0.6)
            axz.set_xlim(zr0, zr1)
            #if i == 0:
            #    axz.set_ylabel('Normalised Flux')
            # Draw connecting lines between main and zoom
            main_ax = axs[f'm{i%3+1}']
            # Fractional x positions in main axis coords
            x0 = (zr0 - windows[i%3][0]) / (windows[i%3][1] - windows[i%3][0])
            x1 = (zr1 - windows[i%3][0]) / (windows[i%3][1] - windows[i%3][0])

            if ax_key in ['z1', 'z2', 'z3']:
                min_x = 1
                max_x = 0
            else:
                min_x = 0
                max_x = 1

            # Connector from left edge of zoom to main
            axz.annotate('', xy=(x0, min_x), xycoords=main_ax.transAxes,
                         xytext=(0, max_x), textcoords=axz.transAxes,
                         arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))
            # Connector from right edge of zoom to main
            axz.annotate('', xy=(x1, min_x), xycoords=main_ax.transAxes,
                         xytext=(1, max_x), textcoords=axz.transAxes,
                         arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

            # vertical lines at zoom edges in the top panel
            main_ax.axvline(zr0, color='gray', linestyle='--', lw=0.5)
            main_ax.axvline(zr1, color='gray', linestyle='--', lw=0.5)

            axz.set_ylim(flux_obs[mask_obs_z].min() - 0.07, flux_obs[mask_obs_z].max() + 0.02)

            # set ticks fontsize
            axz.tick_params(axis='both', which='major', labelsize=11)

            # yticks every 0.1
            #yticks = np.arange(
            #    np.floor(axz.get_ylim()[0] * 10) / 10,
            #    1 + 0.05, 0.2
            #)
            #axz.set_yticks(yticks)

            # set title
            axz.set_title(titles[i], fontsize=12)

            # Highlight specific lines in the zoom panels
            if ax_key in lines_to_highlight:
                for line_name, line_wavelengths in lines_to_highlight[ax_key].items():
                    for wavelength in line_wavelengths:
                        # grey line with a short name on the top
                        ymax = 0.98
                        axz.axvline(wavelength, ymin=0.15, ymax=ymax, color='gray', linestyle='--', lw=0.5)
                        # rotated
                        axz.text(
                            wavelength, 0.02, line_name,
                            color='gray', fontsize=7, ha='center', va='bottom', rotation=75, transform=axz.get_xaxis_transform(
                            )  # use x-axis transform to keep it in the same coordinate system
                        )

        else:
            axz.axis('off')

    axs['z5'].set_xlabel('Wavelength [Å]')

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
        y_base - 0.035,  # a bit below the line
        'Telluric lines in\nobserved spectrum',  # your label
        ha='center', va='top',  # center horizontally, top aligned to that y
        fontsize=10,
        color='gray'
    )

    m2 = axs['m2']

    m2.scatter([-99], [-99], color='black', s=15, label='Observed')
    m2.plot([-99], [-99], color='red', lw=2, label='Payne Fitted')
    m2.legend(fontsize=10, loc='lower right', frameon=False, bbox_to_anchor=(1.1, -0.2))

    # for z2 i need tick seprartion of 5 IT DOESNT WORK
    axs['z2'].xaxis.set_major_locator(plt.MultipleLocator(2))
    axs['z4'].xaxis.set_major_locator(plt.MultipleLocator(0.5))

    #plt.savefig("../plots/HD122563_4MOST-HR_Payne_fit_v2.pdf", bbox_inches='tight')
    plt.show()