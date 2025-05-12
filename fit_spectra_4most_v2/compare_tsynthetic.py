from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 04.04.25

if __name__ == '__main__':
    literature_data = pd.read_csv("/Users/storm/PycharmProjects/payne/ts_nlte_grid_july2024/spectra_parameters.csv")
    payne_data = pd.read_csv("/Users/storm/PycharmProjects/payne/fit_spectra_4most_v2/fitted_250.csv")

    # remove .npy from the file names in payne_data
    payne_data["spectraname"] = payne_data["spectraname"].str.replace("_hrs_cont.npy", "")

    literature_data["spectraname"] = literature_data["specname"]

    # now merge the two dataframes on the spectraname column
    merged_data = pd.merge(literature_data, payne_data, on="spectraname", how="inner")

    print(merged_data.columns)

    cmap = "cool"

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    # ——— Teff ———
    sc0 = ax[0, 0].scatter(
        merged_data["teff_x"],
        merged_data["teff_y"] * 1000,
        c=merged_data["feh_x"],  # colour by feh_x
        cmap=cmap,
        s=14
    )
    ax[0, 0].plot([4000, 8000], [4000, 8000], "r--", color="grey", alpha=0.5)
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
        c=merged_data["feh_x"], cmap=cmap, s=14
    )
    ax[0, 1].plot([0, 5], [0, 5], "r--", color="grey", alpha=0.5)
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
        c=merged_data["feh_x"], cmap=cmap, s=14
    )
    ax[0, 2].plot([-5, 0.5], [-5, 0.5], "r--", color="grey", alpha=0.5)
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
        c=merged_data["feh_x"], cmap=cmap, s=14
    )
    ax[1, 0].plot([0.5, 2], [0.5, 2], "r--", color="grey", alpha=0.5)
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
        c=merged_data["feh_x"], cmap=cmap, s=14
    )
    ax[1, 1].plot([0, 100], [0, 100], "r--", color="grey", alpha=0.5)
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
        c=merged_data["feh_x"], cmap=cmap, s=14
    )
    ax[1, 2].plot([-0.2, 0.2], [-0.2, 0.2], "r--", color="grey", alpha=0.5)
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

    plt.tight_layout()
    plt.show()

    # find how many _Fe_y labels are there
    elements_to_fit = []
    for i, label in enumerate(merged_data.columns):
        if label.endswith("_Fe_y"):
            elements_to_fit.append(label)

    columns = 4
    rows = int(np.ceil(len(elements_to_fit) / columns))

    fig, ax = plt.subplots(rows, columns, figsize=(rows * 3, columns * 2))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, element_to_fit in enumerate(elements_to_fit):
        row = i // columns
        col = i % columns
        min_x = min(list(merged_data[element_to_fit.replace("_y", "_x")] + merged_data["feh_x"]),
                    list(merged_data[element_to_fit] + merged_data["feh_y"]))
        max_x = max(list(merged_data[element_to_fit.replace("_y", "_x")] + merged_data["feh_x"]),
                    list(merged_data[element_to_fit] + merged_data["feh_y"]))

        ax[row, col].plot([min_x, max_x], [min_x, max_x], "--", color="grey", alpha=0.5)

        ax[row, col].scatter(
            merged_data[element_to_fit.replace("_y", "_x")] + merged_data["feh_x"], merged_data[element_to_fit] + merged_data["feh_y"],
            c=merged_data["feh_x"], cmap=cmap, s=14
        )

        ax[row, col].set_xlabel(f"{element_to_fit.replace('_Fe_y', '_H')} (True)")
        ax[row, col].set_ylabel(f"{element_to_fit.replace('_Fe_y', '_H')} (Fitted)")

        ax[row, col].set_title(f"{element_to_fit.replace('_Fe_y', '')}")

    plt.tight_layout()
    plt.show()
