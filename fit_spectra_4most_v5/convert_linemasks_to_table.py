from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 26.08.25

if __name__ == '__main__':
    data = pd.read_csv("all_linemasks.csv")
    # sort by element, then ll
    data = data.sort_values(by=['element', 'll'])
    print(data)

    # convert to format: each column separated by & and each row ends with \\
    with open("all_linemasks_table.txt", "w") as f:
        for index, row in data.iterrows():
            gu = row['gu']
            if pd.isna(gu):
                gu = ""
            else:
                gu = int(gu)
            isotope = row['isotope']
            if pd.isna(isotope):
                isotope = ""
            else:
                isotope = int(isotope)
            if isotope == 0:
                isotope = ""
            elem = row['element']
            wavelength = row['ll']
            if elem == "C2":
                elem = "C$_2$"
                wavelength = int(wavelength)
            elif elem == "CH":
                elem = "CH"
                wavelength = int(wavelength)
            line = f"{elem} & {wavelength:.3f} & {row['elow']:.3f} & {row['loggf']:.3f} & {gu} & {isotope} \\\\"
            f.write(line + "\n")
    print("Saved to all_linemasks_table.txt")