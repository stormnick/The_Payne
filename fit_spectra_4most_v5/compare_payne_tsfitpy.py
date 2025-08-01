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
    payne_data = pd.read_csv("fitted_benchmark_v3.csv")
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
    payne_only.to_csv("payne_only2.csv", index=False)
    #print(payne_only)

    #print(merged_data["source"])
    # only leave merged_data that are ["source"] in ["GES+batch1", "Ruchti"]
    merged_data = merged_data[merged_data["source"].isin(["GES+batch1"])]

    # load tsfitpy data
    main_folder = "/Users/storm/PhD_2025/02.22 Payne/fitted_spectra_tsfitpy/"
    folders = os.listdir(main_folder)
    folders.remove(".DS_Store")
    folders.remove("v1")
    folders.remove("v2")
    folders.remove("extra3")
    #folders.remove("v3")
    tsfitpy_data = pd.DataFrame()
    for folder in folders:
        folder_path = os.path.join(main_folder, folder)
        if os.path.exists(os.path.join(folder_path, "average_abundance.csv")) or os.path.exists(os.path.join(folder_path, "average_abundance_v2.csv")):
            if os.path.exists(os.path.join(folder_path, "average_abundance_v2.csv")):
                new_data = pd.read_csv(os.path.join(folder_path, "average_abundance_v2.csv"))
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
            if element != "Fe_H":
                columns = ["specname", "Fe_H", element, f"{element}_err", "vsini"]
            else:
                columns = ["specname", "Fe_H", f"{element}_err", "vsini"]
            new_data = new_data[columns]
            # rename "vsini" to f"{element}_vsini_tsfitpy"
            if element != "Fe_H":
                new_data.rename(columns={"vsini": f"{element}_vsini", "Fe_H": f"{element}_Fe_H"}, inplace=True)
            else:
                new_data.rename(columns={"vsini": f"{element}_vsini"}, inplace=True)
            if element == "Eu_Fe":
                for row in new_data.iterrows():
                    print(row[1]["specname"], row[1][element], row[1][f"{element}_err"], row[1][f"{element}_vsini"])
            if tsfitpy_data.empty:
                tsfitpy_data = new_data
            else:
                # concat on specname
                tsfitpy_data = pd.merge(tsfitpy_data, new_data, on="specname", how="outer", suffixes=("", "_new"))
    tsfitpy_data.to_csv("tsfitpy_data.csv", index=False)
    print(tsfitpy_data)
    # merge tsfitpy_data with merged_data on spectraname
    tsfitpy_data.rename(columns={"specname": "spectraname_og"}, inplace=True)
    # each column add "_tsfitpy"
    tsfitpy_data.rename(columns=lambda x: f"{x}_tsfitpy" if x not in ["spectraname_og"] else x, inplace=True)
    merged_data = pd.merge(merged_data, tsfitpy_data, on="spectraname_og", how="left", suffixes=("", "_tsfitpy"))
    merged_data["A_Li_tsfitpy"] = merged_data["Li_Fe_tsfitpy"] + merged_data[f"Li_Fe_Fe_H_tsfitpy"] + 1.05
    merged_data.drop(columns=["Li_Fe_tsfitpy"], inplace=True)
    merged_data.to_csv("merged_data_with_tsfitpy_new.csv", index=False)

    # set any merged_data["A_Li"] < 0 to NaN
    merged_data["A_Li_tsfitpy"] = np.where(merged_data["A_Li_tsfitpy"] < 0, np.nan, merged_data["A_Li_tsfitpy"])
    #merged_data["O_Fe"] = np.where(merged_data["O_Fe"] >= 1.25, np.nan, merged_data["O_Fe"])


    if True:
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

        #plt.show()
        plt.close()


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

            print(f"{title}: bias={np.mean(x - y):.3f}, std={np.std(x - y):.3f}")

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

        #plt.show()
        plt.close()

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
             ([3700, 7000], [3700, 7000]), "Teff"),

            (merged_data["logg_x"], merged_data["logg"], merged_data["elogg"],
             ([0.45, 5], [0.45, 5]), "logg"),

            (merged_data["Fe_H_tsfitpy"], merged_data["feh"], merged_data["Fe_H_err_Nick"],
             ([-3.5, 0.32], [-3.5, 0.32]), "[Fe/H]"),
        ]

        # -----------------------------------------------------------
        # 3. plot each panel
        # -----------------------------------------------------------
        for i, (x, y, xerr, (ref_x, ref_y), title) in enumerate(panels):
            ax[i].plot(ref_x, ref_y, "g--", alpha=0.7)  # identity line
            sc = ax[i].scatter(x, y, c='k', s=20)  # points

            # draw coloured error bars one‐by‐one
            for xi, yi, xe, ci in zip(x, y, xerr, colours):
                ax[i].errorbar([xi], [yi], xerr=None, fmt="none",
                               capsize=3, linewidth=0.8, ecolor='k')

            ax[i].set_xlabel(f"{title} (literature)", fontsize=14)
            ax[i].set_ylabel(f"{title} (Payne)", fontsize=14)
            #ax[i].set_title(title, fontsize=14)
            ax[i].set_xlim(ref_x)
            ax[i].set_ylim(ref_y)
        ax[-1].set_xlabel(f"{title} (TSFitPy)")
        # set all x fontsize of ticks
        for i in range(len(ax)):
            ax[i].tick_params(axis='both', which='major', labelsize=14)
        # -----------------------------------------------------------
        # 4. single colour-bar on the right
        # -----------------------------------------------------------
        plt.savefig("../plots/payne_stellar_param_comparison.pdf", bbox_inches='tight')
        plt.show()

    # find how many elements we can plot
    elements_to_plot = []
    for i, label in enumerate(merged_data.columns):
        if label.endswith("_Fe_tsfitpy") or label == "A_Li_tsfitpy":
            elements_to_plot.append(label)
    print(f"Elements to plot: {elements_to_plot}")

    columns = 4
    rows = len(elements_to_plot) // columns + 1
    #rows = 3

    # subplot
    fig, ax = plt.subplots(rows, columns, figsize=(rows * 2, columns * 2), constrained_layout=True)
    ax = ax.flatten()  # flatten the 2D array to 1D for easier indexing
    for i, element in enumerate(elements_to_plot):
        x = np.asarray(merged_data[element])
        y = np.asarray(merged_data[element] - merged_data[element.replace("_tsfitpy", "")])

        x_std = merged_data[f"{element.replace('_tsfitpy', '')}_std"]
        # find any with std < -90, and get their indices
        indices = np.where(x_std < -90)[0]
        # remove any x, y where x_std < -90
        x = np.delete(x, indices)
        y = np.delete(y, indices)
        # find any nan in y, and get their indices
        indices = np.where(np.isnan(y))[0]
        x = np.delete(x, indices)
        y = np.delete(y, indices)

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

    # subplot
    fig, ax = plt.subplots(rows, columns, figsize=(rows * 2, columns * 2), constrained_layout=True)
    ax = ax.flatten()  # flatten the 2D array to 1D for easier indexing
    for i, element in enumerate(elements_to_plot):
        x = np.asarray(merged_data["feh"])
        y = np.asarray(merged_data[element] - merged_data[element.replace("_tsfitpy", "")])

        x_std = merged_data[f"{element.replace('_tsfitpy', '')}_std"]
        # find any with std < -90, and get their indices
        indices = np.where(x_std < -90)[0]
        # remove any x, y where x_std < -90
        x = np.delete(x, indices)
        y = np.delete(y, indices)
        # find any nan in y, and get their indices
        indices = np.where(np.isnan(y))[0]
        x = np.delete(x, indices)
        y = np.delete(y, indices)

        #ax[i].plot([-4, 0.5], [-4, 0.5], "g--")  # identity line
        # horisontal lines at y = 0, -0.2, 0.2
        ax[i].hlines([0, -0.2, 0.2], -3.2, 0.5, colors='k', linestyles='dashed', linewidth=0.8)

        sc = ax[i].scatter(x, y, c='k', s=14)  # points

        ## draw coloured error bars one‐by‐one
        #for xi, yi in zip(x, y):
        #    ax[i].errorbar([xi], [yi], fmt="none",
        #                   capsize=3, linewidth=0.8, ecolor='k')

        ax[i].set_xlabel(f"[Fe/H]")
        ax[i].set_ylabel(f"{rename_element(element)} (TSFitPy - Payne)")
        ax[i].set_title(f"bias={np.mean(y):.3f}, std={np.std(y):.3f}", fontsize=10)

        # ylim
        ax[i].set_ylim(-0.4, 0.4)


    plt.show()


    columns = 4
    rows = len(elements_to_plot) // columns + 1

    # subplot
    fig, ax = plt.subplots(rows, columns, figsize=(rows * 2, columns * 2), constrained_layout=True)
    ax = ax.flatten()  # flatten the 2D array to 1D for easier indexing
    for i, element in enumerate(elements_to_plot):
        x = np.asarray(merged_data["feh"])
        y = np.asarray(merged_data[element] * 0 + merged_data[element.replace("_tsfitpy", "")])
        y2 = np.asarray(merged_data[element]  + merged_data[element.replace("_tsfitpy", "")] * 0)

        x_std = merged_data[f"{element.replace('_tsfitpy', '')}_std"]
        # find any with std < -90, and get their indices
        indices = np.where(x_std < -90)[0]
        # remove any x, y where x_std < -90
        x = np.delete(x, indices)
        y = np.delete(y, indices)
        y2 = np.delete(y2, indices)
        # find any nan in y, and get their indices
        indices = np.where(np.isnan(y))[0]
        x = np.delete(x, indices)
        y = np.delete(y, indices)
        y2 = np.delete(y2, indices)

        #ax[i].plot([-4, 0.5], [-4, 0.5], "g--")  # identity line
        sc = ax[i].scatter(x, y, c='k', s=14)  # points
        sc2 = ax[i].scatter(x, y2, c='r', s=14)

        ## draw coloured error bars one‐by‐one
        #for xi, yi in zip(x, y):
        #    ax[i].errorbar([xi], [yi], fmt="none",
        #                   capsize=3, linewidth=0.8, ecolor='k')

        ax[i].set_xlabel(f"[Fe/H] (Payne)")
        ax[i].set_ylabel(f"{rename_element(element)} (Payne)")
        ax[i].set_title(f"bias={np.mean(y):.3f}, std={np.std(y):.3f}")
        ax[i].set_ylim(-0.5,1)
        ax[i].set_xlim(-3.2, 0.5)

    #plt.savefig("../plots/tsfitpy_payne_gce_comparison.pdf", bbox_inches='tight')
    plt.show()

    columns = 6
    rows = len(elements_to_plot) // columns + 1

    solar_abundances = {
        "H": 12.00,
        "He": 10.93,
        "Li": 1.05,
        "Be": 1.38,
        "B": 2.70,
        "C": 8.56,
        "N": 7.98,
        "O": 8.77,
        "F": 4.40,
        "Ne": 8.16,
        "Na": 6.29,
        "Mg": 7.55,
        "Al": 6.43,
        "Si": 7.59,
        "P": 5.41,
        "S": 7.16,
        "Cl": 5.25,
        "Ar": 6.40,
        "K": 5.14,
        "Ca": 6.37,
        "Sc": 3.07,
        "Ti": 4.94,
        "V": 3.89,
        "Cr": 5.74,
        "Mn": 5.52,
        "Fe": 7.50,
        "Co": 4.95,
        "Ni": 6.24,
        "Cu": 4.19,
        "Zn": 4.56,
        "Ga": 3.04,
        "Ge": 3.65,
        "As": 2.29,  # not in asplund, kept old value
        "Se": 3.33,  # ditto
        "Br": 2.56,  # ditto
        "Kr": 3.25,
        "Rb": 2.52,
        "Sr": 2.87,
        "Y": 2.21,
        "Zr": 2.58,
        "Nb": 1.46,
        "Mo": 1.88,
        "Tc": -99.00,  # ditto
        "Ru": 1.75,
        "Rh": 0.91,
        "Pd": 1.57,
        "Ag": 0.94,
        "Cd": 1.77,  # ditto
        "In": 0.80,
        "Sn": 2.04,
        "Sb": 1.00,  # ditto
        "Te": 2.19,  # ditto
        "I": 1.51,  # ditto
        "Xe": 2.24,
        "Cs": 1.07,  # ditto
        "Ba": 2.18,
        "La": 1.10,
        "Ce": 1.58,
        "Pr": 0.72,
        "Nd": 1.42,
        "Pm": -99.00,  # ditto
        "Sm": 0.96,
        "Eu": 0.52,
        "Gd": 1.07,
        "Tb": 0.30,
        "Dy": 1.10,
        "Ho": 0.48,
        "Er": 0.92,
        "Tm": 0.10,
        "Yb": 0.84,
        "Lu": 0.10,
        "Hf": 0.85,
        "Ta": -0.17,  # ditto
        "W": 0.85,
        "Re": 0.23,  # ditto
        "Os": 1.40,
        "Ir": 1.38,
        "Pt": 1.64,  # ditto
        "Au": 0.92,
        "Hg": 1.13,  # ditto
        "Tl": 0.90,
        "Pb": 1.75,
        "Bi": 0.65,  # ditto
        "Po": -99.00,  # ditto
        "At": -99.00,  # ditto
        "Rn": -99.00,  # ditto
        "Fr": -99.00,  # ditto
        "Ra": -99.00,  # ditto
        "Ac": -99.00,  # ditto
        "Th": 0.02,
        "Pa": -99.00,  # ditto
        "U": -0.52  # ditto
    }

    merged_data["Fe_H"] = merged_data["feh"]  # rename for consistency with Payne data
    merged_data["Fe_H_std"] = merged_data["feh_std"]  # rename for consistency with Payne data

    # order elements_to_plot by the order in solar_abundances
    elements_to_plot = sorted(elements_to_plot, key=lambda x: list(solar_abundances.keys()).index(x.replace("_tsfitpy", "").split("_")[0]) if x.replace("_tsfitpy", "").split("_")[0] in solar_abundances else float('inf'))

    # add Fe_H_tsfitpy to the list of elements to plot, but prelast
    elements_to_plot.insert(-1, "Fe_H_tsfitpy")

    # subplot
    fig, ax = plt.subplots(rows, columns, figsize=(rows * 4, columns * 1), constrained_layout=True)
    ax = ax.flatten()  # flatten the 2D array to 1D for easier indexing
    for i, element in enumerate(elements_to_plot):
        x = np.asarray(merged_data["feh"])

        # find element name in solar_abundances
        if element.replace("_tsfitpy", "") == "A_Li":
            y = np.asarray(merged_data[element] * 0 + merged_data[element.replace("_tsfitpy", "")])  # payne
            x = np.asarray(merged_data[element]  + merged_data[element.replace("_tsfitpy", "")] * 0 ) # tsfitpy
        elif element.replace("_tsfitpy", "") == "Fe_H":
            y = np.asarray(merged_data[element] * 0 + merged_data[element.replace("_tsfitpy", "")])
            x = np.asarray(merged_data[element]  + merged_data[element.replace("_tsfitpy", "")] * 0 )
            solar_abundance_value = solar_abundances["Fe"]
            y = y + solar_abundance_value
            x = x + solar_abundance_value
        else:
            y = np.asarray(merged_data[element] * 0 + merged_data[element.replace("_tsfitpy", "")] + merged_data["feh"])
            x = np.asarray(merged_data[element]  + merged_data[element.replace("_tsfitpy", "")] * 0  + merged_data[f"{element.replace('_tsfitpy', '')}_Fe_H_tsfitpy"])
            solar_abundance_value = solar_abundances[element.replace("_tsfitpy", "").split("_")[0]]
            y = y + solar_abundance_value
            x = x + solar_abundance_value
        x_std = merged_data[f"{element.replace('_tsfitpy', '')}_std"]
        # find any with std < -90, and get their indices
        indices = np.where(x_std < -90)[0]
        # remove any x, y where x_std < -90
        x = np.delete(x, indices)
        y = np.delete(y, indices)
        # find any nan in y, and get their indices
        indices = np.where(np.isnan(y))[0]
        x = np.delete(x, indices)
        y = np.delete(y, indices)

        #ax[i].plot([-4, 0.5], [-4, 0.5], "g--")  # identity line
        #sc = ax[i].scatter(x, y, c='k', s=14)  # points
        sc = ax[i].scatter(x, y, c='k', s=14)

        ## draw coloured error bars one‐by‐one
        #for xi, yi in zip(x, y):
        #    ax[i].errorbar([xi], [yi], fmt="none",
        #                   capsize=3, linewidth=0.8, ecolor='k')

        if element.replace("_tsfitpy", "") == "A_Li":
            ax[i].set_xlabel(f"A(Li) (TSFitPy)")
            ax[i].set_ylabel(f"A(Li) (Payne)")
        elif element.replace("_tsfitpy", "") == "Fe_H":
            ax[i].set_xlabel(f"A(Fe) (TSFitPy)")
            ax[i].set_ylabel(f"A(Fe) (Payne)")
        else:
            ax[i].set_xlabel(f"A({rename_element(element).replace('/Fe]', '').replace('[', '')}) (TSFitPy)")
            ax[i].set_ylabel(f"A({rename_element(element).replace('/Fe]', '').replace('[', '')}) (Payne)")
        #ax[i].set_title(f"bias={np.mean(x - y):.3f}, std={np.std(x - y):.3f}")

        # text in each one with the std
        ax[i].text(0.4, 0.25, f"bias={np.mean(x - y):.3f}\nstd={np.std(x - y):.3f}", transform=ax[i].transAxes,
                   fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.0))

        # draw one-to-one line
        combined = np.concatenate((x, y))

        ax[i].plot([np.min(combined), np.max(combined)], [np.min(combined), np.max(combined)], "g--", alpha=0.7)

        #ax[i].set_ylim(-0.5,1)
        #ax[i].set_xlim(-3.2, 0.5)

    plt.savefig("../plots/a_x_comparison.pdf", bbox_inches='tight')
    plt.show()