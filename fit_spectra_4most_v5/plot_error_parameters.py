from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import fit_systematic_error
import matplotlib.ticker as ticker

matplotlib.use("MacOSX")
plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 29.08.25

if __name__ == '__main__':
    """
    Al_Fe_error
Ba_Fe_error
C_Fe_error 
Ca_Fe_error
Co_Fe_error
Cr_Fe_error
Eu_Fe_error
Mg_Fe_error
Mn_Fe_error
Na_Fe_error
Ni_Fe_error
O_Fe_error 
Si_Fe_error
Sr_Fe_error
Ti_Fe_error
Y_Fe_error 
    """
    errors = [fit_systematic_error.teff_error / 1000, fit_systematic_error.logg_error,
              fit_systematic_error.A_Li_error, fit_systematic_error.C_Fe_error, fit_systematic_error.O_Fe_error,
              fit_systematic_error.Na_Fe_error, fit_systematic_error.Mg_Fe_error, fit_systematic_error.Al_Fe_error,
              fit_systematic_error.Si_Fe_error, fit_systematic_error.Ca_Fe_error, fit_systematic_error.Ti_Fe_error,
              fit_systematic_error.Cr_Fe_error, fit_systematic_error.Mn_Fe_error, fit_systematic_error.feh_error,
              fit_systematic_error.Co_Fe_error,
              fit_systematic_error.Ni_Fe_error, fit_systematic_error.Sr_Fe_error, fit_systematic_error.Y_Fe_error,
              fit_systematic_error.Ba_Fe_error, fit_systematic_error.Eu_Fe_error]
    labels = [r"T$_{\rm eff}$" + "\n" + r"[100K]", "logg", "Li", "C", "O", "Na", "Mg", "Al", "Si",
                "Ca", "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Sr", "Y", "Ba", "Eu"]

    plt.figure(figsize=(6.5, 4))
    plt.scatter(labels, errors, color='k', s=60, zorder=5)
    plt.ylabel("Estimated Systematic Error", fontsize=15)
    plt.xlabel("Label", fontsize=15)
    plt.ylim(0.03, 0.27)
    plt.yscale('log')

    # custom yticks
    yticks = [0.03, 0.05, 0.1, 0.2]
    plt.yticks(yticks, [str(y) for y in yticks], fontsize=15)
    plt.minorticks_off()

    # enlarge x tick labels and alternate positions
    xticks = plt.xticks()[0]  # tick positions
    xlabels = [tick.get_text() for tick in plt.gca().get_xticklabels()]

    plt.gca().set_xticks(xticks)
    new_labels = []
    for i, label in enumerate(xlabels):
        new_labels.append(label)

    # redraw manually with alternating vertical alignment
    for i, label in enumerate(plt.gca().get_xticklabels()):
        label.set_fontsize(15)
        if i % 2 == 0:
            label.set_verticalalignment('top')
            label.set_y(-0.09)  # push down a bit
        else:
            label.set_verticalalignment('bottom')
            label.set_y(-0.07)  # push further down

    plt.tight_layout()
    plt.savefig("../plots/systematic_errors.pdf", bbox_inches='tight')
    plt.show()
