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
    payne_data = pd.read_csv(f"just_elements_fitted_mg_varied_snr.csv")

    # remove .npy from the file names in payne_data
    payne_data["spectraname"] = payne_data["spectraname"].str.split("_hrs_snr", n=1).str[0]

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
    merged_data["A(C)"] = merged_data["C_Fe"] + merged_data["feh_x"] + 8.56
    merged_data["A(O)"] = merged_data["O_Fe"] + merged_data["feh_x"] + 8.77
    # remove too high A(C), A(O) (unrealistic?) and too low feh
    # print length
    print(len(merged_data))
    merged_data = merged_data[merged_data["A(C)"] < 8.56 + 0.2]
    merged_data = merged_data[merged_data["A(O)"] < 8.77 + 0.2]
    merged_data = merged_data[merged_data["feh_x"] >= -4.0]
    print(len(merged_data))

    cmap = "cool"
    
    plot_color_column = merged_data["feh_x"]
    #plot_color_column = merged_data["vsini_x"]
    #plot_color_column = merged_data["N_Fe"]
    #plot_color_column = merged_data["C_Fe_x"]

    snrs_all = pd.unique(merged_data["snr"])
    snrs_all = sorted(snrs_all)

    # find how many _Fe_y labels are there
    element_to_fit = "Mg_Fe_y"

    merged_data["diff"] = merged_data[element_to_fit.replace("_y", "_x")] - merged_data[element_to_fit]
    merged_data_big_diff = merged_data.copy()#[merged_data["diff"] > 0.3]
    merged_data_big_diff = merged_data_big_diff[merged_data_big_diff["snr"] == 1000]
    #merged_data_big_diff.to_csv("diff.csv")

    columns = 3
    rows = int(np.ceil(len(snrs_all) / columns))

    fig, ax = plt.subplots(rows, columns, figsize=(rows * 5, columns * 2))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, snr_one in enumerate(snrs_all):
        merged_data_temp = merged_data[merged_data["snr"] == snr_one]

        row = i // columns
        col = i % columns
        min_x = min(list(merged_data_temp[element_to_fit.replace("_y", "_x")]),
                    list(merged_data_temp[element_to_fit]))
        max_x = max(list(merged_data_temp[element_to_fit.replace("_y", "_x")]),
                    list(merged_data_temp[element_to_fit]))

        ax[row, col].plot([min_x, max_x], [min_x, max_x], "--", color="grey", alpha=0.5)

        x_values = merged_data_temp[element_to_fit.replace("_y", "_x")]
        y_values = merged_data_temp[element_to_fit]
        # now lets check std
        y_values_std = merged_data_temp[f"{element_to_fit.replace('_y', '')}_std"]
        # only choose those with std < 0.2
        std_limit = 0.2
        cond1 = y_values_std < std_limit
        cond2 = y_values_std >= 0
        y_values_good = np.where(cond1 & cond2)[0]
        x_values = np.asarray(x_values)[y_values_good]
        y_values = np.asarray(y_values)[y_values_good]

        ax[row, col].scatter(x_values, y_values, c=np.asarray(plot_color_column)[y_values_good], cmap=cmap, s=14, rasterized=True)

        ax[row, col].set_xlabel(f"{element_to_fit.replace('_Fe_y', '_Fe')} (True)")
        ax[row, col].set_ylabel(f"{element_to_fit.replace('_Fe_y', '_Fe')} (Fitted)")

        diff = x_values - y_values

        ax[row, col].set_title(f"snr={snr_one} len={len(diff)}")
        #ax[row, col].

        print(f'{snr_one:<6} {element_to_fit.replace("_Fe_y", ""):>6}: {np.mean(diff):>6.3f} +/- {np.std(diff):>6.3f} ({len(diff)})')

    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(rows, columns, figsize=(rows * 5, columns * 2))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, snr_one in enumerate(snrs_all):
        merged_data_temp = merged_data[merged_data["snr"] == snr_one]

        row = i // columns
        col = i % columns
        min_x = min(list(merged_data_temp[element_to_fit.replace("_y", "_x")]),
                    list(merged_data_temp[element_to_fit]))
        max_x = max(list(merged_data_temp[element_to_fit.replace("_y", "_x")]),
                    list(merged_data_temp[element_to_fit]))

        ax[row, col].plot([min_x, max_x], [0, 0], "--", color="grey", alpha=0.5)

        x_values = merged_data_temp[element_to_fit.replace("_y", "_x")]
        x_values_plot = merged_data_temp["N_Fe"]
        y_values = merged_data_temp[element_to_fit]
        # now lets check std
        y_values_std = merged_data_temp[f"{element_to_fit.replace('_y', '')}_std"]
        # only choose those with std < 0.2
        std_limit = 0.2
        cond1 = y_values_std < std_limit
        cond2 = y_values_std >= 0
        y_values_good = np.where(cond1 & cond2)[0]
        x_values = np.asarray(x_values)[y_values_good]
        y_values = np.asarray(y_values)[y_values_good]
        x_values_plot = np.asarray(x_values_plot)[y_values_good]

        ax[row, col].scatter(x_values_plot, x_values - y_values, c=np.asarray(plot_color_column)[y_values_good], cmap=cmap, s=6)

        ax[row, col].set_xlabel(f"{element_to_fit.replace('_Fe_y', '_Fe')} (True)")
        ax[row, col].set_xlabel(f"FeH")
        ax[row, col].set_ylabel(f"{element_to_fit.replace('_Fe_y', '_Fe')} (True - Fitted)")

        diff = x_values - y_values

        ax[row, col].set_title(f"snr={snr_one} len={len(diff)}")

        print(f'{snr_one:<6} {element_to_fit.replace("_Fe_y", ""):>6}: {np.mean(diff):>6.3f} +/- {np.std(diff):>6.3f} ({len(diff)})')

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(rows, columns, figsize=(rows * 5, columns * 2))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, snr_one in enumerate(snrs_all):
        merged_data_temp = merged_data[merged_data["snr"] == snr_one]

        row = i // columns
        col = i % columns
        min_x = min(list(merged_data_temp[element_to_fit.replace("_y", "_x")]),
                    list(merged_data_temp[element_to_fit]))
        max_x = max(list(merged_data_temp[element_to_fit.replace("_y", "_x")]),
                    list(merged_data_temp[element_to_fit]))


        # ---------- keep only “good-σ” points ----------
        x_true = merged_data_temp[element_to_fit.replace("_y", "_x")].to_numpy()
        x_feh = merged_data_temp["feh_x"].to_numpy()
        y_fitted = merged_data_temp[element_to_fit].to_numpy()
        y_std = merged_data_temp[f"{element_to_fit.replace('_y', '')}_std"].to_numpy()

        mask = (y_std < 0.2) & (y_std >= 0)
        x_true, x_feh, y_fitted = x_true[mask], x_feh[mask], y_fitted[mask]

        diff = x_true - y_fitted  # what you were plotting on the y-axis

        # ---------- density plot: replace scatter with hexbin ----------
        hb = ax[row, col].hexbin(
            x_feh,  # x-axis (FeH)
            diff,  # y-axis (True – Fitted)
            gridsize=40,  # resolution; larger → finer bins
            cmap="magma",  # pick any perceptually-uniform cmap
            mincnt=1  # do not show empty bins
        )
        cb = fig.colorbar(hb, ax=ax[row, col])
        cb.set_label("objects per bin")

        # ---------- cosmetics ----------
        ax[row, col].set_xlabel("FeH")
        ax[row, col].set_ylabel(f"{element_to_fit.replace('_Fe_y', '_Fe')} (True − Fitted)")
        ax[row, col].set_title(f"S/N = {snr_one}   N = {len(diff)}")
        ax[row, col].plot([min_x, max_x], [0, 0], "--", color="grey", alpha=0.4)

        print(f'{snr_one:<6} {element_to_fit.replace("_Fe_y", ""):>6}: '
              f'{np.mean(diff):>6.3f} ± {np.std(diff):>6.3f} ({len(diff)})')

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(rows, columns, figsize=(rows * 3, columns * 2))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, element_to_fit in enumerate(elements_to_fit):
        row = i // columns
        col = i % columns

        x_values = merged_data[element_to_fit.replace("_y", "_x")]
        x_values_plot = np.asarray(merged_data['teff_x'])[y_values_good]
        y_values = merged_data[element_to_fit]
        # only choose those with std < 0.2
        x_values = np.asarray(x_values)[y_values_good]
        y_values = np.asarray(y_values)[y_values_good]
        #x_values_plot = x_values

        min_x = min(x_values_plot)
        max_x = max(x_values_plot)

        # horisontal line
        ax[row, col].plot([min_x, max_x], [0, 0], "--", color="grey", alpha=0.5)

        ax[row, col].scatter(x_values_plot,
            x_values - y_values,
            c=np.asarray(merged_data['feh_x'])[y_values_good], cmap=cmap, s=14
        )

        ax[row, col].set_xlabel(f"{element_to_fit.replace('_Fe_y', '_Fe')} (True)")
        ax[row, col].set_ylabel(f"{element_to_fit.replace('_Fe_y', '_Fe')} (True - Fitted)")

        ax[row, col].set_title(f"{element_to_fit.replace('_Fe_y', '')}")
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
