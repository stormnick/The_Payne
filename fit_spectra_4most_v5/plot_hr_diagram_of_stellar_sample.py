from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 08.08.25


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
    plt.scatter(data["teff"] * 1000, data["logg"], c=data["feh"], cmap="viridis", s=50, edgecolor='none')
    plt.colorbar(label="[Fe/H]")  # colorbar label to [Fe/H]
    # colorbar fontsize to 15

    plt.xlabel(r"T$_{\rm eff}$ [K]")
    plt.ylabel("logg [dex]")
    plt.xlim(3550, 7000)
    plt.ylim(0.2, 4.9)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    # set fontsize fo both axes
    #plt.tick_params(axis='both', which='major')
    plt.savefig("../plots/hr_diagram_stellar_sample.pdf", dpi=300)
    plt.show()