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
    payne_data = pd.read_csv("fitted_values_melchiors_v2.csv")

    # remove .npy from the file names in payne_data
    payne_data["spectraname"] = payne_data["spectraname"].str.replace(".npy", "")

    literature_data["spectraname"] = literature_data["HD"]

    # now merge the two dataframes on the spectraname column
    merged_data = pd.merge(literature_data, payne_data, on="spectraname", how="inner")

    print(merged_data.columns)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].plot([4000, 8000], [4000, 8000], "r--")
    ax[0].scatter(merged_data["Teff"], merged_data["teff"] * 1000, color='k', label="Teff", s=14)
    ax[0].errorbar(merged_data["Teff"], merged_data["teff"] * 1000, markersize=0, xerr=merged_data["e_Teff"], linestyle="None", color='k', capsize=3)
    ax[0].set_xlabel("Teff (Soubiran)")
    ax[0].set_ylabel("Teff (Payne)")
    ax[0].set_title("Teff")

    ax[1].plot([0, 5], [0, 5], "r--")
    ax[1].scatter(merged_data["logg_x"], merged_data["logg_y"], color='k', label="logg", s=14)
    ax[1].errorbar(merged_data["logg_x"], merged_data["logg_y"], markersize=0, xerr=merged_data["e_logg"], linestyle="None", color='k', capsize=3)
    ax[1].set_xlabel("logg (Soubiran)")
    ax[1].set_ylabel("logg (Payne)")
    ax[1].set_title("logg")

    ax[2].plot([-3, 0.5], [-3, 0.5], "r--")
    ax[2].scatter(merged_data["[Fe/H]"], merged_data["feh"], color='k', label="feh", s=14)
    ax[2].errorbar(merged_data["[Fe/H]"], merged_data["feh"], markersize=0, xerr=merged_data["e_[Fe/H]"], linestyle="None", color='k', capsize=3)
    ax[2].set_xlabel("[Fe/H] (Soubiran)")
    ax[2].set_ylabel("[Fe/H] (Payne)")
    ax[2].set_title("[Fe/H]")
    plt.savefig("compare_melchiors_melchiors.png")
    plt.show()
