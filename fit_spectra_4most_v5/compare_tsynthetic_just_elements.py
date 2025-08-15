from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 04.04.25

if __name__ == '__main__':
    literature_data = pd.read_csv("/Users/storm/PycharmProjects/payne/ts_nlte_grid_apr2024/spectra_parameters_nlte_batch0_v3.csv")
    literature_data["A_Li"] = literature_data["Li_Fe"] + 1.05 + literature_data["feh"]
    snr_to_do = 50
    #payne_data = pd.read_csv(f"fitted_synthetic_just_elements.csv")
    payne_data = pd.read_csv(f"fitted_synthetic_just_elements_{snr_to_do}.csv")

    # remove .npy from the file names in payne_data
    payne_data["spectraname"] = payne_data["spectraname"].str.replace(f"_hrs_snr{snr_to_do}.0.npy", "").str.replace("/Users/storm/PycharmProjects/payne/ts_nlte_grid_apr2024/batch0_nlte_4mostified_v3/", "")

    bad_spectra_literature = pd.read_csv("/Users/storm/PycharmProjects/payne/ts_nlte_grid_apr2024/bad_spectra.csv")

    # build a Boolean mask: keep rows whose specname is **not** in bad_spectra_literature
    good_mask = ~literature_data['specname'].isin(bad_spectra_literature['specname'])

    # apply it (and make a copy so you don’t get chained-assignment warnings later)
    literature_data = literature_data[good_mask]

    literature_data["spectraname"] = literature_data["specname"]

    # build a Boolean mask: keep rows whose specname is **not** in bad_spectra_literature
    good_mask = ~literature_data['specname'].isin(bad_spectra_literature['specname'])

    # apply it (and make a copy so you don’t get chained-assignment warnings later)
    literature_data_clean = literature_data[good_mask].copy()

    # now merge the two dataframes on the spectraname column
    merged_data = pd.merge(literature_data, payne_data, on="spectraname", how="inner")

    print(merged_data.columns)

    # Create column A(C)
    merged_data["A(C)"] = merged_data["C_Fe_x"] + merged_data["feh_x"] + 8.56
    merged_data["A(O)"] = merged_data["O_Fe_x"] + merged_data["feh_x"] + 8.77
    # remove too high A(C), A(O) (unrealistic?) and too low feh
    # print length
    print(len(merged_data))
    merged_data = merged_data[merged_data["A(C)"] < 8.7]
    merged_data = merged_data[merged_data["A(O)"] < 8.87]
    merged_data = merged_data[merged_data["feh_x"] >= -5.0]
    merged_data = merged_data[merged_data["logg_x"] >= 0.5]
    print(len(merged_data))

    cmap = "cool"
    
    plot_color_column = merged_data["feh_x"]
    #plot_color_column = merged_data["vsini_x"]
    #plot_color_column = merged_data["N_Fe"]
    #plot_color_column = merged_data["C_Fe_x"]

    """fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    # ——— Teff ———
    sc0 = ax[0, 0].scatter(
        merged_data["teff_x"],
        merged_data["teff_y"] * 1000,
        c=plot_color_column,  # colour by feh_x
        cmap=cmap,
        s=14
    )
    ax[0, 0].plot([4000, 8000], [4000, 8000], "--", color="grey", alpha=0.5)
    ax[0, 0].errorbar(
        merged_data["teff_x"], merged_data["teff_y"] * 1000,
        markersize=0, linestyle="None",
        ecolor="lightgray", capsize=3
    )
    ax[0, 0].set_xlabel("Teff (True)")
    ax[0, 0].set_ylabel("Teff (Fitted)")
    ax[0, 0].set_title("Teff")

    # ——— log g ———
    ax[0, 1].scatter(
        merged_data["logg_x"], merged_data["logg_y"],
        c=plot_color_column, cmap=cmap, s=14
    )
    ax[0, 1].plot([0, 5], [0, 5], "--", color="grey", alpha=0.5)
    ax[0, 1].errorbar(
        merged_data["logg_x"], merged_data["logg_y"],
        markersize=0, linestyle="None",
        ecolor="lightgray", capsize=3
    )
    ax[0, 1].set_xlabel("log g (True)")
    ax[0, 1].set_ylabel("log g (Fitted)")
    ax[0, 1].set_title("log g")

    # ——— [Fe/H] ———
    ax[0, 2].scatter(
        merged_data["feh_x"], merged_data["feh_y"],
        c=plot_color_column, cmap=cmap, s=14
    )
    ax[0, 2].plot([-5, 0.5], [-5, 0.5], "--", color="grey", alpha=0.5)
    ax[0, 2].errorbar(
        merged_data["feh_x"], merged_data["feh_y"],
        markersize=0, linestyle="None",
        ecolor="lightgray", capsize=3
    )
    ax[0, 2].set_xlabel("[Fe/H] (True)")
    ax[0, 2].set_ylabel("[Fe/H] (Fitted)")
    ax[0, 2].set_title("[Fe/H]")


    # --- vmic ---
    ax[1, 0].scatter(
        merged_data["vmic_x"], merged_data["vmic_y"],
        c=plot_color_column, cmap=cmap, s=14
    )
    ax[1, 0].plot([0.5, 2], [0.5, 2], "--", color="grey", alpha=0.5)
    ax[1, 0].errorbar(
        merged_data["vmic_x"], merged_data["vmic_y"],
        markersize=0, linestyle="None",
        ecolor="lightgray", capsize=3
    )
    ax[1, 0].set_xlabel("vmic (True)")
    ax[1, 0].set_ylabel("vmic (Fitted)")
    ax[1, 0].set_title("vmic")
    # --- vsini ---
    ax[1, 1].scatter(
        merged_data["vsini_x"], merged_data["vsini_y"],
        c=plot_color_column, cmap=cmap, s=14
    )
    ax[1, 1].plot([0, 100], [0, 100], "--", color="grey", alpha=0.5)
    ax[1, 1].errorbar(
        merged_data["vsini_x"], merged_data["vsini_y"],
        markersize=0, linestyle="None",
        ecolor="lightgray", capsize=3
    )
    ax[1, 1].set_xlabel("vsini (True)")
    ax[1, 1].set_ylabel("vsini (Fitted)")
    ax[1, 1].set_title("vsini")
    # log x and y
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_yscale("log")
    # --- rv ---
    ax[1, 2].scatter(
        len(list(merged_data["doppler_shift"])) * [0], merged_data["doppler_shift"],
        c=plot_color_column, cmap=cmap, s=14
    )
    ax[1, 2].plot([-0.2, 0.2], [-0.2, 0.2], "--", color="grey", alpha=0.5)
    ax[1, 2].errorbar(
        len(list(merged_data["doppler_shift"])) * [0], merged_data["doppler_shift"],
        markersize=0, linestyle="None",
        ecolor="lightgray", capsize=3
    )
    ax[1, 2].set_xlabel("rv (True)")
    ax[1, 2].set_ylabel("rv (Fitted)")


    # Shared colour-bar
    #cbar = fig.colorbar(sc0, ax=ax.ravel().tolist()[-1], location="right")
    #cbar.set_label("[Fe/H] true")

    teff_diff = merged_data["teff_x"] - merged_data["teff_y"] * 1000
    print(f'Teff: {np.mean(teff_diff):>6.3f} +/- {np.std(teff_diff):>6.3f} ({len(teff_diff)})')
    logg_diff = merged_data["logg_x"] - merged_data["logg_y"]
    print(f'logg: {np.mean(logg_diff):>6.3f} +/- {np.std(logg_diff):>6.3f} ({len(logg_diff)})')
    feh_diff = merged_data["feh_x"] - merged_data["feh_y"]
    print(f'feh: {np.mean(feh_diff):>6.3f} +/- {np.std(feh_diff):>6.3f} ({len(feh_diff)})')
    vmic_diff = merged_data["vmic_x"] - merged_data["vmic_y"]
    print(f'vmic: {np.mean(vmic_diff):>6.3f} +/- {np.std(vmic_diff):>6.3f} ({len(vmic_diff)})')
    vsini_diff = merged_data["vsini_x"] - merged_data["vsini_y"]
    print(f'vsini: {np.mean(vsini_diff):>6.3f} +/- {np.std(vsini_diff):>6.3f} ({len(vsini_diff)})')

    plt.tight_layout()
    plt.show()"""

    # find how many _Fe_y labels are there
    elements_to_fit = []
    for i, label in enumerate(merged_data.columns):
        if label.endswith("_Fe_y") or label == "A_Li_y":
            elements_to_fit.append(label)

    columns = 6
    rows = int(np.ceil(len(elements_to_fit) / columns))

    fig, ax = plt.subplots(rows, columns, figsize=(rows * 4, columns * 1))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    for i, element_to_fit in enumerate(elements_to_fit):
        row = i // columns
        col = i % columns
        min_x = min(list(merged_data[element_to_fit.replace("_y", "_x")]),
                    list(merged_data[element_to_fit]))
        max_x = max(list(merged_data[element_to_fit.replace("_y", "_x")]),
                    list(merged_data[element_to_fit]))

        ax[row, col].plot([min(min_x), max(max_x)], [min(min_x), max(max_x)], "--", color="grey", alpha=0.5)

        x_values = merged_data[element_to_fit.replace("_y", "_x")]
        y_values = merged_data[element_to_fit]
        # now lets check std
        y_values_std = merged_data[f"{element_to_fit.replace('_y', '')}_std"]
        # only choose those with std < 0.2
        std_limit = 0.3
        cond1 = y_values_std < std_limit
        cond2 = y_values_std >= 0
        y_values_good = np.where(cond1 & cond2)[0]
        x_values = np.asarray(x_values)[y_values_good]
        y_values = np.asarray(y_values)[y_values_good]

        ax[row, col].scatter(x_values, y_values, c=np.asarray(plot_color_column)[y_values_good], cmap=cmap, s=14)

        # text inside with the element
        ax[row, col].text(0.05, 0.95, element_to_fit.replace("_Fe_y", "").replace("A_Li_y", "Li"), transform=ax[row, col].transAxes,
                            fontsize=12, verticalalignment='top', horizontalalignment='left')

        #ax[row, col].set_xlabel(f"True")
        #ax[row, col].set_ylabel(f"Fitted")

        #ax[row, col].set_title(f"{element_to_fit.replace('_Fe_y', '')}")

        diff = x_values - y_values

        print(f'{element_to_fit.replace("_Fe_y", ""):>6}: {np.mean(diff):>6.3f} +/- {np.std(diff):>6.3f} ({len(diff)})')

    plt.tight_layout()
    plt.show()

    values_to_compare = ["teff_x", "logg_x", "feh_x", "vmic_x", "vsini_x", "C_Fe_x", "O_Fe_x", "N_Fe"]

    for value in values_to_compare:

        columns = 6
        rows = int(np.ceil(len(elements_to_fit) / columns))

        fig, ax = plt.subplots(rows, columns, figsize=(rows * 4, columns * 1))
        fig.subplots_adjust(hspace=0.0, wspace=0.0)

        for i, element_to_fit in enumerate(elements_to_fit):
            row = i // columns
            col = i % columns

            x_values = merged_data[element_to_fit.replace("_y", "_x")]
            x_values_plot = np.asarray(merged_data[value])[y_values_good]
            y_values = merged_data[element_to_fit]
            # only choose those with std < 0.2
            x_values = np.asarray(x_values)[y_values_good]
            y_values = np.asarray(y_values)[y_values_good]
            #x_values_plot = x_values

            try:
                min_x = min(x_values_plot)
                max_x = max(x_values_plot)

                # horisontal line
                ax[row, col].plot([min_x, max_x], [0, 0], "--", color="grey", alpha=0.5)
            except ValueError:
                pass

            #ax[row, col].scatter(x_values_plot,
            #    x_values - y_values,
            #    c=np.asarray(merged_data['feh_x'])[y_values_good], cmap=cmap, s=14
            #)
            ax[row, col].scatter(x_values_plot,
                x_values - y_values,s=14, c='k'
            )

            ax[row, col].text(0.05, 0.95, element_to_fit.replace("_Fe_y", "").replace("A_Li_y", "Li"),
                              transform=ax[row, col].transAxes,
                              fontsize=12, verticalalignment='top', horizontalalignment='left')

            ax[row, col].set_xlabel(f"{value} (True)")
            #ax[row, col].set_ylabel(f"{element_to_fit.replace('_Fe_y', '_Fe')} (True - Fitted)")

            #ax[row, col].set_title(f"{element_to_fit.replace('_Fe_y', '')}")
            ax[row, col].set_ylim(-1, 1)

        plt.tight_layout()
        plt.show()

    fig, ax = plt.subplots(rows, columns,
                           figsize=(columns * 3.5, rows * 2.8))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)

    # ---------- pre-compute residuals & global limits ----------
    residuals = {}
    gmin, gmax = np.inf, -np.inf

    for el in elements_to_fit:
        res = (merged_data[el.replace('_y', '_x')] + merged_data['feh_x']
               ) - (merged_data[el] + merged_data['feh_y'])
        residuals[el] = res
        gmin, gmax = min(gmin, res.min()), max(gmax, res.max())

    lim = max(abs(gmin), abs(gmax))  # symmetric about zero
    xlims = (-lim, lim)
    xlims = (-1, 1)

    # ---------- plotting ----------
    for i, el in enumerate(elements_to_fit):
        r = residuals[el]
        row, col = divmod(i, columns)
        ax_i = ax[row, col]

        # Freedman-Diaconis rule for bin width
        q75, q25 = np.percentile(r, [75, 25])
        iqr = q75 - q25
        bw = 2 * iqr * len(r) ** (-1 / 3) if iqr else None
        nbins = int((xlims[1] - xlims[0]) * 30) #if bw else 30
        #nbins = 50

        # histogram
        ax_i.hist(r, bins=nbins, color='steelblue',
                  alpha=0.7, edgecolor='white')

        # reference & summary lines
        ax_i.axvline(0, color='grey', ls='--', lw=1)
        μ, σ, med = r.mean(), r.std(ddof=1), np.median(r)
        ax_i.axvline(μ, color='darkorange', ls='-', lw=1.5)
        ax_i.axvline(med, color='firebrick', ls=':', lw=1.5)
        ax_i.fill_betweenx([0, ax_i.get_ylim()[1]], μ - σ, μ + σ,
                           color='darkorange', alpha=0.15)

        # cosmetics
        ax_i.set_xlim(*xlims)
        ax_i.set_xlabel(f"{el.replace('_Fe_y', '_H')}  (True − Fitted)  [dex]")
        ax_i.set_title(el.replace('_Fe_y', ''))
        ax_i.legend([f"σ = {σ:.3f}", f"μ = {μ:+.3f}", f"median = {med:+.3f}"],
                    fontsize='x-small', frameon=False)

        ax_i.set_xlim((-1, 1))

    plt.tight_layout()
    plt.show()
