from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import fit_systematic_error
import matplotlib.ticker as ticker

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

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
    #plt.xlabel("Estimated Systematic Error (dex or K for Teff)")
    #plt.title("Estimated Systematic Errors in Stellar Parameters and Abundances")
    #plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.ylabel("Estimated Systematic Error")
    plt.xlabel("Label")
    plt.ylim(0.03, 0.26)
    # change yticks to 0.03, 0.05, 0.1, 0.2
    plt.yscale('log')
    # REMOVE OTHER TICKS BEFOREHAND
    plt.yticks([0.03, 0.05, 0.1, 0.2], ["0.03", "0.05", "0.1", "0.2"])
    plt.minorticks_off()  # disables all minor ticks
    plt.tight_layout()
    plt.savefig("../plots/systematic_errors.pdf", bbox_inches='tight')
    plt.show()