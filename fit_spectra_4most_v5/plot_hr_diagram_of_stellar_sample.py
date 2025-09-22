from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import fit_systematic_error

matplotlib.use("MacOSX")
plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 08.08.25

def convert_to_logg(model):
    logg = convert_to_teff_logg(model["LOG_TE"], model["LOG_L"], model["MASS"])
    model["logg"] = logg
    return model

    MSUN = 1.988e30  # kg
    RSUN = 6.957e8  # m
    G = 6.67430e-11  # m^3 kg^-1
    g = G * model["MASS"] * MSUN / (10**model["LOG_R"] / 100) ** 2 * 100
    logg = np.log10(g)
    model["logg"] = logg
    return model

def convert_to_teff_logg(logt, logl, mass):
    SUNMg = 1.9891e+33            #Sun mass in grams
    SUNLe = 3.846e+33             #Sun luminosity in erg/s
    SUNRcm = 6.95508e10           #Sun radius in cm
    PI = 3.1415926535898e0
    GRAVC0 = 6.672320e-8
    STBOLZ=5.6704e-5
    SUNMv = 4.75
    teff = 10 ** logt
    lum = 10 ** logl
    logg = np.log10((4 * PI * STBOLZ * GRAVC0) * (mass * SUNMg) * (teff ** 4) / (lum * SUNLe))
    return logg


if __name__ == '__main__':
    data = pd.read_csv("fitted_benchmark_batch012_v2.csv")

    # remove rows where spectraname == "NARVAL_61CygB.txt" or "NARVAL_61CygA.txt"
    data = data[~data["spectraname"].isin(["NARVAL_61CygB.txt", "NARVAL_61CygA.txt"])]

    # print ranges of teff, logg, vmic, feh
    print(f"T_eff: {data['teff'].min()} - {data['teff'].max()}")
    print(f"logg: {data['logg'].min()} - {data['logg'].max()}")
    print(f"vmic: {data['vmic'].min()} - {data['vmic'].max()}")
    print(f"[Fe/H]: {data['feh'].min()} - {data['feh'].max()}")

    # print how many with logg > 4.1
    print(f"Number of stars with logg >= 4.1: {len(data[data['logg'] >= 4.1])}")
    # how any 3.5 <= logg < 4.1
    print(f"Number of stars with 3.5 <= logg < 4.1: {len(data[(data['logg'] >= 3.5) & (data['logg'] < 4.1)])}")
    # how many with logg < 3.5
    print(f"Number of stars with logg < 3.5: {len(data[data['logg'] < 3.5])}")

    plt.figure()

    models = ["/Users/storm/Downloads/Z0.014Y0.273/Z0.014Y0.273OUTA1.74_F7_M000.800.DAT",
              "/Users/storm/Downloads/Z0.014Y0.273/Z0.014Y0.273OUTA1.74_F7_M001.000.DAT",
              "/Users/storm/Downloads/Z0.0001Y0.249/Z0.0001Y0.249OUTA1.74_F7_M000.800.DAT",
              "/Users/storm/Downloads/Z0.0001Y0.249/Z0.0001Y0.249OUTA1.74_F7_M001.000.DAT"]
    names = [r"[M/H] = 0, $0.8 M_{\odot}$",
             r"[M/H] = 0, $1.0 M_{\odot}$",
             r"[M/H] = -2, $0.8 M_{\odot}$",
             r"[M/H] = -2, $1.0 M_{\odot}$"]
    colors = ['r', 'r', 'b', 'b']
    linestyles = ['-', '--', '-', '--']

    for model, name, color, linestyle in zip(models, names, colors, linestyles):
        garstec_model = pd.read_csv(model, sep='\s+', header=0)
        garstec_model = convert_to_logg(garstec_model)
        garstec_model = garstec_model[garstec_model["PHASE"] > 4]
        plt.plot(10**garstec_model["LOG_TE"], garstec_model["logg"], c=color, label=name, linewidth=2, linestyle=linestyle)



    #plt.scatter(data["teff"] * 1000, data["logg"], c=data["feh"], cmap="viridis", s=50, edgecolor='none')
    plt.scatter(data["teff"] * 1000, data["logg"], c='k', s=50, edgecolor='none')
    # add errorbars
    plt.errorbar(data["teff"] * 1000, data["logg"], xerr=fit_systematic_error.teff_error + data["teff_std"] * 1000, yerr=fit_systematic_error.logg_error + data["logg_std"], fmt='none', ecolor='k', elinewidth=0.5, capsize=1)
    #plt.colorbar(label="[Fe/H]")  # colorbar label to [Fe/H]
    # colorbar fontsize to 15

    plt.xlabel(r"T$_{\rm eff}$ [K]")
    plt.ylabel("logg")
    plt.xlim(3100, 8200)
    plt.ylim(-0.2, 5.0)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    # set fontsize fo both axes
    #plt.tick_params(axis='both', which='major')
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(-0.03, 1.03))
    plt.savefig("../plots/hr_diagram_stellar_sample.pdf", dpi=300)
    plt.show()