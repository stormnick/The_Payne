from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 04.04.25

if __name__ == '__main__':
    literature_data = pd.read_csv("../literature_param.csv")
    payne_data = pd.read_csv("fitted_values_melchiors_v2_test.csv")

    # remove .npy from the file names in payne_data
    payne_data["spectraname"] = payne_data["spectraname"].str.replace(".npy", "")

    literature_data["spectraname"] = literature_data["HD"]

    # now merge the two dataframes on the spectraname column
    merged_data = pd.merge(literature_data, payne_data, on="spectraname", how="inner")

    print(merged_data.columns)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # -----------------------------------------------------------
    # 1. common colour map
    # -----------------------------------------------------------
    cvals = merged_data["[Fe/H]"] - merged_data["feh"]  # value we colour by
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(cvals.min(), cvals.max())
    colours = cmap(norm(cvals))  # RGBA for every star

    # -----------------------------------------------------------
    # 2. helper list to drive the three panels
    # -----------------------------------------------------------
    panels = [
        #  x                      y                     x-err                1:1 line        title
        (merged_data["Teff"], merged_data["teff"] * 1000, merged_data["e_Teff"],
         ([4000, 8000], [4000, 8000]), "Teff"),

        (merged_data["logg_x"], merged_data["logg_y"], merged_data["e_logg"],
         ([0, 5], [0, 5]), "log g"),

        (merged_data["[Fe/H]"], merged_data["feh"], merged_data["e_[Fe/H]"],
         ([-3, 0.5], [-3, 0.5]), "[Fe/H]"),
    ]

    # -----------------------------------------------------------
    # 3. plot each panel
    # -----------------------------------------------------------
    for i, (x, y, xerr, (ref_x, ref_y), title) in enumerate(panels):
        ax[i].plot(ref_x, ref_y, "g--")  # identity line
        sc = ax[i].scatter(x, y, c=cvals, cmap=cmap, norm=norm, s=14)  # points

        # draw coloured error bars one‐by‐one
        for xi, yi, xe, ci in zip(x, y, xerr, colours):
            ax[i].errorbar([xi], [yi], xerr=[xe], fmt="none",
                           ecolor=ci, capsize=3, linewidth=0.8)

        ax[i].set_xlabel(f"{title} (Soubiran)")
        ax[i].set_ylabel(f"{title} (Payne)")
        ax[i].set_title(title)

    # -----------------------------------------------------------
    # 4. single colour-bar on the right
    # -----------------------------------------------------------
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax.ravel().tolist(),
        location="right",
        label=r"$\Delta\mathrm{[Fe/H]}\;(\mathrm{Soubiran}-\mathrm{Payne})$"
    )

    plt.show()

