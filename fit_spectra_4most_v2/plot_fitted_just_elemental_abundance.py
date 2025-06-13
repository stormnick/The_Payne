from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from payne_fit_clean_full import load_payne, plot_fitted_payne
from spectral_model import get_spectrum_from_neural_net
import os

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 15.05.25

if __name__ == '__main__':
    literature_data = pd.read_csv("/Users/storm/PycharmProjects/payne/ts_nlte_grid_apr2024/spectra_parameters_nlte_batch0_v3.csv")
    snr_to_do = 1000
    payne_data = pd.read_csv(f"just_elements_fitted_{snr_to_do}.csv")

    # remove .npy from the file names in payne_data
    payne_data["spectraname"] = payne_data["spectraname"].str.replace(f"_hrs_snr{snr_to_do}.0.npy", "").str.replace("/Users/storm/PycharmProjects/payne/ts_nlte_grid_apr2024/batch0_nlte_4mostified_v3/", "")

    bad_spectra_literature = pd.read_csv("/Users/storm/PycharmProjects/payne/ts_nlte_grid_apr2024/bad_spectra.csv")

    # build a Boolean mask: keep rows whose specname is **not** in bad_spectra_literature
    good_mask = ~literature_data['specname'].isin(bad_spectra_literature['specname'])

    # apply it (and make a copy so you don’t get chained-assignment warnings later)
    literature_data = literature_data[good_mask]

    literature_data["spectraname"] = literature_data["specname"]

    # build a Boolean mask: keep rows whose specname is **not** in bad_spectra_literature
    good_mask = ~literature_data['specname'].isin(bad_spectra_literature['specname'])

    # apply it (and make a copy so you don’t get chained-assignment warnings later)
    literature_data_clean = literature_data[good_mask].copy()

    # now merge the two dataframes on the spectraname column
    merged_data = pd.merge(literature_data, payne_data, on="spectraname", how="inner")

    print(merged_data.columns)

    # Create column A(C)
    merged_data["A(C)"] = merged_data["C_Fe_x"] + merged_data["feh_x"] + 8.56
    merged_data["A(O)"] = merged_data["O_Fe_x"] + merged_data["feh_x"] + 8.77
    # remove too high A(C), A(O) (unrealistic?) and too low feh
    # print length
    print(len(merged_data))
    merged_data = merged_data[merged_data["A(C)"] < 8.56 + 0.2]
    merged_data = merged_data[merged_data["A(O)"] < 8.77 + 0.2]
    merged_data = merged_data[merged_data["feh_x"] >= -4.0]
    print(len(merged_data))

    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_2025-03-27-08-06-34.npz"

    payne_coeffs, wavelength_payne, labels = load_payne(path_model)
    x_min = list(payne_coeffs[-2])
    x_max = list(payne_coeffs[-1])

    label_names = labels.copy()
    label_names.append('vsini')
    label_names.append('vmac')
    label_names.append('doppler_shift')

    # get all filenames in the folder
    snr_to_do = 1000
    folder_spectra = f"/Users/storm/PycharmProjects/payne/ts_nlte_grid_july2024/snr{int(snr_to_do)}/"
    folder_spectra = f"/Users/storm/PhD_2025/spectra_4most/batch0_nlte_4mostified_v3/"
    files = os.listdir(folder_spectra)
    # only take those with "hrs" in it
    files = [file for file in files if f"hrs_snr{snr_to_do}" in file]

    fitted_values = pd.DataFrame(columns=["spectraname", *labels, "vsini", "vmac", "doppler_shift"])

    parameter_to_check = "Mg_Fe"
    lines = pd.read_csv("../linemasks/mg.csv")['ll'].tolist()
    #lines = [x for x in lines if x > 6000]
    dlam = 1
    ylims = [0.2, 1.02]
    
    for row in merged_data.iterrows():
        if row[1]['spectraname'] != "19236.spec":
            continue
        print(row[1]['spectraname'])
        for value in row[1].items():
            print(f"{value[0]}: {value[1]}")

        real_labels = {}
        for label in labels:
            if label not in parameter_to_check:
                if label in row[1].keys():
                    real_labels[label] = (row[1][label])
                else:
                    real_labels[label] = row[1][label + "_x"]
            else:
                real_labels[label] = (row[1][label + "_y"])

        real_labels['teff'] = real_labels['teff'] / 1000

        real_labels['vsini'] = row[1]['vsini_x']
        real_labels['vmac'] = 0
        real_labels['doppler_shift'] = 0

        file_to_fit = f"{str(row[1]['spectraname'])}_hrs_snr{snr_to_do}.0.npy"

        continuum = np.load(f"{folder_spectra}/{file_to_fit.replace(f'snr{snr_to_do}.0', f'cont')}")
        spectra = np.load(f"{folder_spectra}/{file_to_fit}")
        wavelength_obs = continuum[0]
        flux_obs = spectra[1] / continuum[1]

        for line in lines:
            plot_fitted_payne(
                wavelength_payne,
                real_labels,
                payne_coeffs,
                wavelength_obs,
                flux_obs,
                labels, plot_show=False,) #  real_labels2=[4.9112, 3.526, -2.32, 1.62], real_labels2_xfe={"C_Fe": 1.528, "Ca_Fe": 0.751,
                                                                                                   #   "Ba_Fe": 0.46, "O_Fe": -0.5,
                                                                                                   #   "Mn_Fe": -0.007, "Co_Fe": 1,
                                                                                                   #   "Sr_Fe": 1, "Eu_Fe": 2, "Mg_Fe": 0.773,
                                                                                                   #   "Ti_Fe": 0.35, "Y_Fe": 0.399}, real_labels2_vsini=0.28
            plt.title(f"{row[1]['spectraname']}, {row[1]['teff_x']}/{row[1]['logg_x']}/{row[1]['feh_x']}/{row[1]['vmic_x']:.2f}/{row[1]['vsini_x']:.2f}, {parameter_to_check} true = {row[1][f'{parameter_to_check}_x']:.2f}, fitted = {row[1][f'{parameter_to_check}_y']:.2f}")
            xlims = [line - dlam, line + dlam]
            plt.xlim(xlims)
            plt.ylim(ylims)
            plt.axvline(x=line, color='r', linestyle='--', label=f"{parameter_to_check} = {line:.2f}")
            plt.ylim(0.5, 1.01)
            plt.show()
