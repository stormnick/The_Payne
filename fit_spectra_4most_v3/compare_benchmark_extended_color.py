from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 04.04.25

if __name__ == '__main__':
    literature_data_benchmark = pd.read_csv("/Users/storm/PycharmProjects/payne/observed_spectra_to_test/Table1_updated_S24C25.csv")
    payne_data = pd.read_csv("fitted_benchmark_extended.csv")
    literature_data_ruchti = pd.read_csv("/Users/storm/PycharmProjects/payne/june25_test/ruchti2013_literature_lte.csv")

    # identify rows that begin with "j" or "t"  (already lower-case in your file)
    mask_jt = payne_data['spectraname'].str.startswith(('j', 't'))

    # split on the first dot and keep the part before it
    payne_data.loc[mask_jt, 'spectraname'] = (
        payne_data.loc[mask_jt, 'spectraname'].str.split('.', n=1).str[0]
    )

    # remove .npy from the file names in payne_data
    payne_data["spectraname"] = payne_data["spectraname"].str.replace(".txt", "").str.replace("IWG7_", "")

    literature_data_benchmark["origin"] = literature_data_benchmark["origin"].str.replace("BS_", "")
    literature_data_benchmark["origin"] = literature_data_benchmark["origin"].str.replace("UVES-POP", "UVESPOP")
    literature_data_benchmark["origin"] = literature_data_benchmark["origin"].str.replace("-", "")
    literature_data_benchmark["origin"] = literature_data_benchmark["origin"].str.replace("UVESPOP", "UVES-POP")

    literature_data_benchmark["spectraname"] = np.where(
        literature_data_benchmark["origin"].astype(bool),  # True if origin is non-empty / non-NaN
        literature_data_benchmark["origin"] + "_" + literature_data_benchmark["star"],
        literature_data_benchmark["star"]  # if origin is empty → just star
    )

    literature_data_ruchti["spectraname"] = literature_data_ruchti["Name"]

    r_mask = literature_data_ruchti['Name'].str.startswith('R')  # RAVE …
    t_mask = literature_data_ruchti['Name'].str.startswith('T')  # TYC  …

    # -------- RAVE  →  jxxxx  -----------------------------------------------
    literature_data_ruchti.loc[r_mask, 'Name'] = (
        literature_data_ruchti.loc[r_mask, 'Name']  # original strings
        .str.replace(r'^RAVE\s+', '', regex=True)  # drop leading “RAVE ”
        .str.replace('.', '', regex=False)  # remove every dot
        .str.replace(r'^J', 'j', regex=True)  # capital J → j
    )

    # -------- TYC   →  txxxx  -----------------------------------------------
    literature_data_ruchti.loc[t_mask, 'Name'] = (
            't' +  # add the leading “t”
            literature_data_ruchti.loc[t_mask, 'Name']
            .str.replace(r'^TYC\s+', '', regex=True)  # drop leading “TYC ”
            .str.slice(stop=-2)  # trim last two chars”
    )

    literature_data_ruchti.loc[t_mask, 'Name'] = literature_data_ruchti.loc[t_mask, 'Name'].str.rstrip('f')

    literature_data_ruchti["spectraname"] = literature_data_ruchti["Name"]

    # combine into one literature_data
    literature_data = pd.concat([literature_data_benchmark, literature_data_ruchti], ignore_index=True)

    # now merge the two dataframes on the spectraname column
    merged_data = pd.merge(literature_data, payne_data, on="spectraname", how="inner")
    mask = ~payne_data['spectraname'].isin(literature_data['spectraname'])
    payne_only = payne_data[mask]  # <-- the rows you’re after
    print(payne_only)


    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # -----------------------------------------------------------
    # 1. common colour map
    # -----------------------------------------------------------
    cvals = merged_data["feh_x"] - merged_data["feh_y"]  # value we colour by
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(cvals.min(), cvals.max())
    colours = cmap(norm(cvals))  # RGBA for every star

    # -----------------------------------------------------------
    # 2. helper list to drive the three panels
    # -----------------------------------------------------------
    panels = [
        #  x                      y                     x-err                1:1 line        title
        (merged_data["Teff"], merged_data["teff"] * 1000, merged_data["eTeff"],
         ([4000, 8000], [4000, 8000]), "Teff"),

        (merged_data["logg_x"], merged_data["logg_y"], merged_data["elogg"],
         ([0, 5], [0, 5]), "logg"),

        (merged_data["feh_x"], merged_data["feh_y"], merged_data["efeh"],
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


    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # -----------------------------------------------------------
    # 1. common colour map
    # -----------------------------------------------------------
    cvals = merged_data["Fe_H_Nick"] - merged_data["feh_y"]  # value we colour by
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(cvals.min(), cvals.max())
    colours = cmap(norm(cvals))  # RGBA for every star

    # -----------------------------------------------------------
    # 2. helper list to drive the three panels
    # -----------------------------------------------------------
    panels = [
        #  x                      y                     x-err                1:1 line        title
        (merged_data["Teff"], merged_data["teff"] * 1000, merged_data["eTeff"],
         ([4000, 8000], [4000, 8000]), "Teff"),

        (merged_data["logg_x"], merged_data["logg_y"], merged_data["elogg"],
         ([0, 5], [0, 5]), "logg"),

        (merged_data["Fe_H_Nick"], merged_data["feh_y"], merged_data["Fe_H_err_Nick"],
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
    ax[-1].set_xlabel(f"{title} (TSFitPy)")

    # -----------------------------------------------------------
    # 4. single colour-bar on the right
    # -----------------------------------------------------------
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax.ravel().tolist(),
        location="right",
        label=r"$\Delta\mathrm{[Fe/H]}\;(\mathrm{TSFitPy}-\mathrm{Payne})$"
    )

    plt.show()

