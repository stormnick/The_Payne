from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 08.08.25


if __name__ == '__main__':
    data = pd.read_csv("merged_data_with_tsfitpy_new.csv")

    red_curve = pd.read_csv("extracted_red.csv")
    blue_curve = pd.read_csv("extracted_blue.csv")
    green_curve = pd.read_csv("extracted_green.csv")
    magenta_curve = pd.read_csv("extracted_magenta.csv")
    black_curve = pd.read_csv("extracted_black.csv")
    colors = ["black", "red", "blue", "green", "magenta", ]
    curves = [black_curve, red_curve, blue_curve, green_curve, magenta_curve]
    curve_labels = ["Li total", r"$^{6}$Li", r"$^{7}$Li primordial", r"$^{7}$Li GCR", r"$^{7}$Li $\nu$-process"]
    linestyles = ['-', ':', '--', (5, (10, 3)), '-.']

    # in each add A(Li) column
    for curve in curves:
        curve["Li_H"] = np.log10(curve["Li_over_H"]) + 12
        curve["A_Li"] = curve["Li_H"]
        curve["Li_Fe"] = curve["Li_H"] - 1.05 - curve["Fe_H"]

    # test curve
    for color, curve in zip(colors, curves):
        plt.plot(curve["Fe_H"], curve["Li_over_H"], linestyle='--', c=color, label=f"{color.capitalize()} curve")
    plt.xlabel("[Fe/H]")
    plt.ylabel("Li/H")
    plt.yscale('log')
    plt.xlim(-4, 0.5)
    plt.ylim(1e-12, 2.2e-9)
    plt.legend()
    #plt.show()
    plt.close()

    # remove rows where spectraname == "NARVAL_61CygB.txt" or "NARVAL_61CygA.txt"
    data = data[~data["spectraname"].isin(["NARVAL_61CygB.txt", "NARVAL_61CygA.txt"])]

    element = "A_Li_tsfitpy"

    ali_payne = np.asarray(data[element] * 0 + data[element.replace("_tsfitpy", "")])  # payne

    # indices where ali_payne is not nan
    idx = np.where(~np.isnan(ali_payne))[0]
    ali_payne = ali_payne[idx]
    feh_plot = np.asarray(data["feh"])[idx]  # feh
    xerr_plot = np.asarray(data["feh_std"])[idx]  # error in feh
    yerr_plot = np.asarray(data[f"A_Li_std"])[idx]  # error

    fontsize = 14

    # 2 subplots: A(Li) and Li/Fe
    fig, axs = plt.subplots(1, 1, figsize=(6.5, 6), sharex=True)
    axs.scatter(feh_plot, ali_payne, s=40, edgecolor='none', c='k')
    axs.errorbar(feh_plot, ali_payne, xerr=xerr_plot, yerr=yerr_plot, fmt='none', ecolor='k', elinewidth=1, capsize=3)
    axs.set_ylabel(r"A(Li)", fontsize=fontsize)
    axs.set_xlim(-4.0, 0.4)
    axs.set_ylim(0.0, 3.5)
    axs.set_xlabel(r"[Fe/H]", fontsize=fontsize)
    #axs[0].axhline(1.05, color='black', linestyle='--', linewidth=0.8)
    for color, curve, label, linestyle in zip(colors, curves, curve_labels, linestyles):
        axs.plot(curve["Fe_H"], curve["A_Li"], label=label, color=color, linestyle=linestyle, lw=2)
    #axs[0].axhline(y=2.72, color='lightgrey', linestyle='--',linewidth=3, alpha=0.5)
    #axs[0].text(-3.1, 2.9, f"Primordial value",
    #                   fontsize=11, verticalalignment='top', fontfamily='sans-serif',
    #                   horizontalalignment='left', color='grey')

    axs.legend(loc='upper left', fontsize=14, frameon=False, ncol=2, #bbox_to_anchor=(-0.03, 1.02),
                     handlelength=3, handletextpad=0.3, columnspacing=0.0,
                    labelspacing=0.15)

    # change all fontsizes
    axs.tick_params(axis='both', which='major', labelsize=fontsize)
    axs.tick_params(axis='both', which='minor', labelsize=fontsize)

    plt.tight_layout()
    plt.savefig("../plots/Li_abundance_vs_Fe.pdf", bbox_inches='tight')
    plt.show()