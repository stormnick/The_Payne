from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 04.04.25

if __name__ == '__main__':
    literature_data = pd.read_csv("/Users/storm/Downloads/2025-06-02-08-18-31_0.12270961457306906_NLTE_Mg_1D/average_abundance.csv")
    payne_data = pd.read_csv("fitted_benchmark.csv")

    # remove .npy from the file names in payne_data
    #payne_data["spectraname"] = payne_data["spectraname"].str.replace(".npy", "")

    literature_data["spectraname"] = literature_data["specname"]

    # now merge the two dataframes on the spectraname column
    merged_data = pd.merge(literature_data, payne_data, on="spectraname", how="inner")
    merged_data.to_csv("123tw.csv", index=False)
    mask = ~payne_data['spectraname'].isin(literature_data['spectraname'])
    payne_only = payne_data[mask]  # <-- the rows you’re after
    print(payne_only)


    fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)

    x, y, (ref_x, ref_y), title = merged_data["Mg_Fe_x"], merged_data["Mg_Fe_y"], ([-0.2, 0.6], [-0.2, 0.6]), "[Mg/Fe]"
    ax.plot(ref_x, ref_y, "g--")  # identity line
    sc = ax.scatter(x, y, c='k',s=14)  # points

    ## draw coloured error bars one‐by‐one
    #for xi, yi, xe, ci in zip(x, y, xerr, colours):
    #    ax[i].errorbar([xi], [yi], xerr=[xe], fmt="none",
    #                   ecolor=ci, capsize=3, linewidth=0.8)

    ax.set_xlabel(f"{title} (TSFitPy)")
    ax.set_ylabel(f"{title} (Payne)")
    ax.set_title(title)
    ax.set_xlabel(f"{title} (TSFitPy)")

    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)

    ax.scatter(merged_data["Fe_H"],  merged_data["Mg_Fe_x"], c='k',s=14, label='Payne')  # points
    ax.scatter(merged_data["Fe_H"],  merged_data["Mg_Fe_y"], c='r',s=14, label='TSFitPy')  # points

    ## draw coloured error bars one‐by‐one
    #for xi, yi, xe, ci in zip(x, y, xerr, colours):
    #    ax[i].errorbar([xi], [yi], xerr=[xe], fmt="none",
    #                   ecolor=ci, capsize=3, linewidth=0.8)

    ax.set_xlabel(f"[Fe/H]")
    ax.set_ylabel(f"[Mg/Fe]")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)

    ax.scatter(merged_data["Fe_H"],  merged_data["Mg_Fe_x"] - merged_data["Mg_Fe_y"], c='k',s=14)  # points

    ## draw coloured error bars one‐by‐one
    #for xi, yi, xe, ci in zip(x, y, xerr, colours):
    #    ax[i].errorbar([xi], [yi], xerr=[xe], fmt="none",
    #                   ecolor=ci, capsize=3, linewidth=0.8)

    ax.set_xlabel(f"[Fe/H]")
    ax.set_ylabel(f"[Mg/Fe] Payne - TSFitPy")
    plt.show()

