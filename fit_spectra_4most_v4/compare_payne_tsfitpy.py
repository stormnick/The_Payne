from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.colors import ListedColormap
import os

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 04.04.25

def rename_element(element):
    element = element.replace("_tsfitpy", "")
    if element == "A_Li":
        return "A(Li)"
    elif element == "Fe_H":
        return "[Fe/H]"
    else:
        return f"[{element.split('_')[0]}/{element.split('_')[1]}]"

if __name__ == '__main__':
    payne_data = pd.read_csv("fitted_benchmark_extended_refitted.csv")
    literature_data_benchmark = pd.read_csv("/Users/storm/PycharmProjects/payne/observed_spectra_to_test/Table1_updated_S24C25.csv")
    literature_data_benchmark["source"] = "GES+batch1"
    literature_data_ruchti = pd.read_csv("/Users/storm/PycharmProjects/payne/june25_test/ruchti2013_literature_lte.csv")
    literature_data_ruchti["source"] = "Ruchti"

    payne_data['spectraname_og'] = payne_data['spectraname']  # keep original names for later

    # identify rows that begin with "j" or "t"  (already lower-case in your file)
    mask_jt = payne_data['spectraname'].str.startswith(('j', 't'))

    # split on the first dot and keep the part before it
    payne_data.loc[mask_jt, 'spectraname'] = (
        payne_data.loc[mask_jt, 'spectraname'].str.split('.', n=1).str[0]
    )

    mask_tf = (
            payne_data['spectraname'].str.startswith(('j', 't')) &
            payne_data['spectraname'].str.endswith('f')
    )
    payne_data.loc[mask_tf, 'spectraname'] = (
        payne_data.loc[mask_tf, 'spectraname'].str.rstrip('f')  # or .str[:-1]
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

    literature_data_ruchti["spectraname"] = literature_data_ruchti["Name"]

    # combine into one literature_data
    literature_data = pd.concat([literature_data_benchmark, literature_data_ruchti], ignore_index=True)

    # now merge the two dataframes on the spectraname column
    merged_data = pd.merge(literature_data, payne_data, on="spectraname", how="inner", suffixes=("_x", ""))
    #merged_data.to_csv("merged_data.csv", index=False)
    mask = ~payne_data['spectraname'].isin(literature_data['spectraname'])
    payne_only = payne_data[mask]  # <-- the rows you’re after
    #payne_only.to_csv("payne_only.csv", index=False)
    #print(payne_only)

    #print(merged_data["source"])
    # only leave merged_data that are ["source"] in ["GES+batch1", "Ruchti"]
    #merged_data = merged_data[merged_data["source"].isin(["GES+batch1"])]

    # load tsfitpy data
    main_folder = "/Users/storm/PhD_2025/02.22 Payne/fitted_spectra_tsfitpy/"
    folders = os.listdir(main_folder)
    folders.remove(".DS_Store")
    tsfitpy_data = pd.DataFrame()
    for folder in folders:
        folder_path = os.path.join(main_folder, folder)
        if os.path.exists(os.path.join(folder_path, "average_abundance.csv")):
            if os.path.exists(os.path.join(folder_path, "average_abundance_2.csv")):
                new_data = pd.read_csv(os.path.join(folder_path, "average_abundance_2.csv"))
            else:
                new_data = pd.read_csv(os.path.join(folder_path, "average_abundance.csv"))
            columns = new_data.columns
            element = ""
            for col in columns:
                if col.endswith("_Fe"):
                    element = col
                    break
            if element == "":
                element = "Fe_H"
            columns = ["specname", element, f"{element}_err"]
            new_data = new_data[columns]
            if tsfitpy_data.empty:
                tsfitpy_data = new_data
            else:
                # concat on specname
                tsfitpy_data = pd.merge(tsfitpy_data, new_data, on="specname", how="outer", suffixes=("", "_new"))
    tsfitpy_data.to_csv("tsfitpy_data.csv", index=False)

    # merge tsfitpy_data with merged_data on spectraname
    tsfitpy_data.rename(columns={"specname": "spectraname_og"}, inplace=True)
    # each column add "_tsfitpy"
    tsfitpy_data.rename(columns=lambda x: f"{x}_tsfitpy" if x not in ["spectraname_og"] else x, inplace=True)
    merged_data = pd.merge(merged_data, tsfitpy_data, on="spectraname_og", how="left", suffixes=("", "_tsfitpy"))
    merged_data["A_Li_tsfitpy"] = merged_data["Li_Fe_tsfitpy"] + merged_data["feh"] + 1.05
    merged_data.drop(columns=["Li_Fe_tsfitpy"], inplace=True)
    merged_data.to_csv("merged_data_with_tsfitpy.csv", index=False)

    if False:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

        # -----------------------------------------------------------
        # 1. common colour map
        # -----------------------------------------------------------
        cvals = merged_data["feh_x"] - merged_data["feh"]  # value we colour by
        cmap = plt.cm.coolwarm
        norm = plt.Normalize(cvals.min(), cvals.max())
        colours = cmap(norm(cvals))  # RGBA for every star

        # -----------------------------------------------------------
        # 2. helper list to drive the three panels
        # -----------------------------------------------------------
        panels = [
            #  x                      y                     x-err                1:1 line        title
            (merged_data["Teff"], merged_data["teff"] * 1000, merged_data["eTeff"],
             ([3800, 8000], [3800, 8000]), "Teff"),

            (merged_data["logg_x"], merged_data["logg"], merged_data["elogg"],
             ([0, 5], [0, 5]), "logg"),

            (merged_data["feh_x"], merged_data["feh"], merged_data["efeh"],
             ([-3.2, 0.5], [-3.2, 0.5]), "[Fe/H]"),
        ]

        # -----------------------------------------------------------
        # 3. plot each panel
        # -----------------------------------------------------------
        for i, (x, y, xerr, (ref_x, ref_y), title) in enumerate(panels):
            ax[i].plot(ref_x, ref_y, "g--")  # identity line

            codes, uniques = merged_data["source"].factorize()
            cmap = ListedColormap(["0", "tab:red", "tab:green"][:len(uniques)])

            sc = ax[i].scatter(x, y, c=codes, cmap=cmap, s=10)

            # draw coloured error bars one‐by‐one
            #for xi, yi, xe, ci in zip(x, y, xerr, colours):
            #    ax[i].errorbar([xi], [yi], xerr=[xe], fmt="none",
            #                   ecolor=ci, capsize=3, linewidth=0.8)

            ax[i].set_xlabel(f"{title} (Soubiran)")
            ax[i].set_ylabel(f"{title} (Payne)")
            ax[i].set_title(title)

        handles, labels = sc.legend_elements(prop="colors")
        # replace numeric labels with original category names
        ax[-1].legend(handles, uniques)

        # -----------------------------------------------------------
        # 4. single colour-bar on the right
        # -----------------------------------------------------------
        #fig.colorbar(
        #    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        #    ax=ax.ravel().tolist(),
        #    location="right",
        #    label=r"$\Delta\mathrm{[Fe/H]}\;(\mathrm{Soubiran}-\mathrm{Payne})$"
        #)

        plt.show()


        fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

        # -----------------------------------------------------------
        # 1. common colour map
        # -----------------------------------------------------------
        cvals = merged_data["Fe_H_tsfitpy"] - merged_data["feh"]  # value we colour by
        cmap = plt.cm.coolwarm
        norm = plt.Normalize(cvals.min(), cvals.max())
        colours = cmap(norm(cvals))  # RGBA for every star

        # -----------------------------------------------------------
        # 2. helper list to drive the three panels
        # -----------------------------------------------------------
        panels = [
            #  x                      y                     x-err                1:1 line        title
            (merged_data["Teff"], merged_data["teff"] * 1000, merged_data["eTeff"],
             ([3800, 8000], [3800, 8000]), "Teff"),

            (merged_data["logg_x"], merged_data["logg"], merged_data["elogg"],
             ([0, 5], [0, 5]), "logg"),

            (merged_data["Fe_H_tsfitpy"], merged_data["feh"], merged_data["Fe_H_err_Nick"],
             ([-3.2, 0.5], [-3.2, 0.5]), "[Fe/H]"),
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
                               capsize=3, linewidth=0.8)

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

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

        # -----------------------------------------------------------
        # 1. common colour map
        # -----------------------------------------------------------
        cvals = merged_data["Fe_H_tsfitpy"] - merged_data["feh"]  # value we colour by
        cmap = plt.cm.coolwarm
        norm = plt.Normalize(cvals.min(), cvals.max())
        colours = cmap(norm(cvals))  # RGBA for every star

        # -----------------------------------------------------------
        # 2. helper list to drive the three panels
        # -----------------------------------------------------------
        panels = [
            #  x                      y                     x-err                1:1 line        title
            (merged_data["Teff"], merged_data["teff"] * 1000, merged_data["eTeff"],
             ([3800, 8000], [3800, 8000]), "Teff"),

            (merged_data["logg_x"], merged_data["logg"], merged_data["elogg"],
             ([0, 5], [0, 5]), "logg"),

            (merged_data["Fe_H_tsfitpy"], merged_data["feh"], merged_data["Fe_H_err_Nick"],
             ([-3.2, 0.5], [-3.2, 0.5]), "[Fe/H]"),
        ]

        # -----------------------------------------------------------
        # 3. plot each panel
        # -----------------------------------------------------------
        for i, (x, y, xerr, (ref_x, ref_y), title) in enumerate(panels):
            ax[i].plot(ref_x, ref_y, "g--")  # identity line
            sc = ax[i].scatter(x, y, c='k', s=14)  # points

            # draw coloured error bars one‐by‐one
            for xi, yi, xe, ci in zip(x, y, xerr, colours):
                ax[i].errorbar([xi], [yi], xerr=None, fmt="none",
                               capsize=3, linewidth=0.8, ecolor='k')

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

    # find how many elements we can plot
    elements_to_plot = []
    for i, label in enumerate(merged_data.columns):
        if label.endswith("_Fe_tsfitpy") or label == "A_Li_tsfitpy":
            elements_to_plot.append(label)
    print(f"Elements to plot: {elements_to_plot}")

    columns = 4
    rows = len(elements_to_plot) // columns + 1
    rows = 3

    # subplot
    fig, ax = plt.subplots(rows, columns, figsize=(rows * 3, columns * 2), constrained_layout=True)
    ax = ax.flatten()  # flatten the 2D array to 1D for easier indexing
    for i, element in enumerate(elements_to_plot):
        x = merged_data[element]
        y = merged_data[element] - merged_data[element.replace("_tsfitpy", "")]

        #ax[i].plot([-4, 0.5], [-4, 0.5], "g--")  # identity line
        sc = ax[i].scatter(x, y, c='k', s=14)  # points

        ## draw coloured error bars one‐by‐one
        #for xi, yi in zip(x, y):
        #    ax[i].errorbar([xi], [yi], fmt="none",
        #                   capsize=3, linewidth=0.8, ecolor='k')

        ax[i].set_xlabel(f"{rename_element(element)} (TSFitPy)")
        ax[i].set_ylabel(f"{rename_element(element)} (TSFitPy - Payne)")
        ax[i].set_title(f"bias={np.mean(y):.3f}, std={np.std(y):.3f}")


    plt.show()

    columns = 4
    rows = len(elements_to_plot) // columns + 1
    rows = 3

    # subplot
    fig, ax = plt.subplots(rows, columns, figsize=(rows * 3, columns * 2), constrained_layout=True)
    ax = ax.flatten()  # flatten the 2D array to 1D for easier indexing
    for i, element in enumerate(elements_to_plot):
        x = merged_data["feh"]
        y = merged_data[element] - merged_data[element.replace("_tsfitpy", "")]

        #ax[i].plot([-4, 0.5], [-4, 0.5], "g--")  # identity line
        sc = ax[i].scatter(x, y, c='k', s=14)  # points

        ## draw coloured error bars one‐by‐one
        #for xi, yi in zip(x, y):
        #    ax[i].errorbar([xi], [yi], fmt="none",
        #                   capsize=3, linewidth=0.8, ecolor='k')

        ax[i].set_xlabel(f"[Fe/H] (Payne)")
        ax[i].set_ylabel(f"{rename_element(element)} (TSFitPy - Payne)")
        ax[i].set_title(f"bias={np.mean(y):.3f}, std={np.std(y):.3f}")


    plt.show()