from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 29.07.25


if __name__ == '__main__':
    data = pd.read_csv("merged_data_with_tsfitpy_new.csv")

    # ok g through the columns: "feh", "vmic", "vsini", "vmac", "doppler_shift",
    #         "C_Fe", "O_Fe", "Al_Fe", "Ba_Fe", "Ca_Fe", "Co_Fe", "Cr_Fe", "Eu_Fe",
    #         "Mg_Fe", "Mn_Fe", "Na_Fe", "Ni_Fe", "Si_Fe", "Sr_Fe", "Ti_Fe", "Y_Fe",
    #         "A_Li"
    columns_total = ["feh",
            "C_Fe", "O_Fe", "Al_Fe", "Ba_Fe", "Ca_Fe", "Co_Fe", "Cr_Fe", "Eu_Fe",
             "Mg_Fe", "Mn_Fe", "Na_Fe", "Ni_Fe", "Si_Fe", "Sr_Fe", "Ti_Fe", "Y_Fe",
            "A_Li"]

    # rename Fe_H_tsfitpy to feh_tsfitpy
    data.rename(columns={"Fe_H_tsfitpy": "feh_tsfitpy"}, inplace=True)

    mask = data[[f"{c}_tsfitpy" for c in columns_total]].isna()
    data[columns_total] = data[columns_total].where(~mask)

    # only want to keep the following columns: star,spectraname,teff,logg,feh,vmic,vsini,vmac,doppler_shift,C_Fe,O_Fe,Al_Fe,Ba_Fe,Ca_Fe,Co_Fe,Cr_Fe,Eu_Fe,Mg_Fe,Mn_Fe,Na_Fe,Ni_Fe,Si_Fe,Sr_Fe,Ti_Fe,Y_Fe,A_Li,teff_std,logg_std,feh_std,vmic_std,vsini_std,vmac_std,doppler_shift_std,C_Fe_std,O_Fe_std,Al_Fe_std,Ba_Fe_std,Ca_Fe_std,Co_Fe_std,Cr_Fe_std,Eu_Fe_std,Mg_Fe_std,Mn_Fe_std,Na_Fe_std,Ni_Fe_std,Si_Fe_std,Sr_Fe_std,Ti_Fe_std,Y_Fe_std,A_Li_std
    data = data[[
        "star", "spectraname", "teff", "logg", "feh", "vmic", "vsini", "vmac", "doppler_shift",
        "C_Fe", "O_Fe", "Al_Fe", "Ba_Fe", "Ca_Fe", "Co_Fe", "Cr_Fe", "Eu_Fe",
        "Mg_Fe", "Mn_Fe", "Na_Fe", "Ni_Fe", "Si_Fe", "Sr_Fe", "Ti_Fe", "Y_Fe",
        "A_Li",
        "teff_std", "logg_std", "feh_std", "vmic_std", "vsini_std", "vmac_std",
        "doppler_shift_std",
        "C_Fe_std", "O_Fe_std", "Al_Fe_std", "Ba_Fe_std", "Ca_Fe_std",
        "Co_Fe_std", "Cr_Fe_std", "Eu_Fe_std",
        "Mg_Fe_std", "Mn_Fe_std", "Na_Fe_std", "Ni_Fe_std",
        "Si_Fe_std", "Sr_Fe_std", "Ti_Fe_std",
        "Y_Fe_std",
        'A_Li_std'
    ]]

    std_cols = [f"{c}_std" for c in columns_total]

    # Boolean mask: True where the corresponding _std value is < –90
    mask = data[std_cols].lt(-90)

    # Apply the mask to the value columns
    data[columns_total] = data[columns_total].where(~mask)
    data[std_cols] = data[std_cols].where(~mask)

    print(data)

    data.to_csv("merged_data_with_tsfitpy_new_cleaned.csv", index=False)