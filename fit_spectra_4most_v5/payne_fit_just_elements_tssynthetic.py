from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy
from The_Payne import spectral_model
from convolve import *
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from payne_fit_clean_full import (fit_stellar_parameters, fit_one_xfe_element, process_spectra, load_payne,
                                  plot_fitted_payne, PayneParams, StellarParameters, create_default_stellar_parameters)
import random
import os
from tqdm.contrib.concurrent import process_map
from itertools import repeat
from time import perf_counter

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 03.03.25

def fit_one_spectrum(file, stellar_rv, folder, payne_parameters, snr_to_do, true_parameter):
    print(f"Fitting {file}")
    continuum = np.load(f"{file.replace(f'snr{snr_to_do}.0', f'cont')}")
    spectra = np.load(f"{file}")
    wavelength_obs = continuum[0]
    flux_obs = spectra[1] / continuum[1]
    h_line_cores = pd.read_csv("../linemasks/h_cores.csv")
    h_line_cores = list(h_line_cores['ll'])
    wavelength_obs, flux_obs = process_spectra(wavelength_payne, wavelength_obs, flux_obs, h_line_cores,
                                               h_line_core_mask_dlam=0.2)
    stellar_parameters = create_default_stellar_parameters(payne_parameters)
    stellar_parameters.teff = true_parameter['teff'] / 1000
    stellar_parameters.logg = true_parameter['logg']
    stellar_parameters.feh = true_parameter['feh']
    stellar_parameters.vmic = true_parameter['vmic']
    stellar_parameters.vsini = true_parameter['vsini']
    stellar_parameters.vmac = 0

    elements_to_fit = []
    for i, label in enumerate(labels):
        if label.endswith("_Fe") or label == "A_Li":
            elements_to_fit.append(label)

    priority = {"C_Fe": 0, "O_Fe": 1}  # everything else gets 2 by default
    elements_to_fit = sorted(
        (lbl for lbl in elements_to_fit),
        key=lambda x: (priority.get(x, 2), x)  # (first by priority, then alphabetically)
    )

    for element_to_fit in elements_to_fit:
        stellar_parameters = fit_one_xfe_element(element_to_fit, stellar_parameters, payne_parameters, wavelength_obs,
                                                 flux_obs, silent=True, sigma_flux=1 / snr_to_do)

    if "/" in file:
        filename_to_save = file.split("/")[-1]
    else:
        filename_to_save = file

    final_parameters = {lab: p.value for lab, p in stellar_parameters.iter_params()}
    final_parameters_std = {lab: p.std for lab, p in stellar_parameters.iter_params()}

    print(f"Fitted parameters for {filename_to_save}:")
    print(stellar_parameters)

    # add to fitted_values
    new_row_df = pd.DataFrame(
        [[filename_to_save, *final_parameters.values(), *final_parameters_std.values()]],
        columns=["spectraname"] + list(final_parameters.keys()) + [f"{name}_std" for name in final_parameters.keys()]
    )

    return new_row_df

def _wrapper(file, folder, payne_parameters, snr_to_do, true_parameter):
    stellar_rv = 0
    return fit_one_spectrum(file, stellar_rv, folder, payne_parameters, snr_to_do, true_parameter)


if __name__ == '__main__':
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_batch012_medium_reducedlogg_altarch_notscratch_2025-08-08-11-33-02.npz"
    payne_coeffs, wavelength_payne, labels = load_payne(path_model)
    x_min = list(payne_coeffs[-2])
    x_max = list(payne_coeffs[-1])

    label_names = labels.copy()
    label_names.append('vsini')
    label_names.append('vmac')
    label_names.append('doppler_shift')

    stellar_rv = 0
    resolution_val = None

    # get all filenames in the folder
    snr_to_do = 1000
    folder_spectra = f"/Users/storm/PycharmProjects/payne/ts_nlte_grid_july2024/snr{int(snr_to_do)}/"
    folder_spectra = f"/Users/storm/PhD_2025/spectra_4most/batch0_nlte_4mostified_v3/"
    # only take those with "hrs" in it
    # grab every filename ending in ".npy"
    all_files = [f for f in os.listdir(folder_spectra) if f.endswith(".npy")]

    ids = sorted({f.split('.')[0] for f in all_files})

    random.seed(42)
    chosen_ids = set(random.sample(ids, 50))

    true_values = pd.read_csv("/Users/storm/PycharmProjects/payne/ts_nlte_grid_apr2024/spectra_parameters_nlte_batch0_v3.csv")
    true_values["spectraname"] = true_values["specname"]

    def get_spectra(snr_to_do: int):
        tag = f"hrs_snr{snr_to_do}"
        return [
            os.path.join(folder_spectra, f)
            for f in all_files
            if tag in f and f.split('.')[0] in chosen_ids
        ]

    files = get_spectra(snr_to_do)
    print(f"Fitting {len(files)} spectra with SNR {snr_to_do}")

    true_parameters = []

    for file in files:
        # find that row in true_values["spectraname"]
        if "/" in file:
            filename_to_save = file.split("/")[-1]
        else:
            filename_to_save = file
        row_df = true_values[true_values["spectraname"] == filename_to_save.replace(f"_hrs_snr{snr_to_do}.0.npy", "")]
        true_parameters.append(row_df.to_dict(orient="records")[0])

    payne_parameters = PayneParams(
        payne_coeffs=payne_coeffs,
        wavelength_payne=wavelength_payne,
        labels=labels,
        x_min=x_min,
        x_max=x_max,
        resolution_val=resolution_val
    )

    # file, folder, payne_parameters, all_snr_df, true_parameter

    rows = process_map(
        _wrapper,
        files,  # iterable #1  → `path`
        repeat(folder_spectra),  # iterable #2  → `folder` (reused for every item)
        repeat(payne_parameters),
        repeat(snr_to_do),
        true_parameters,
        max_workers=8,
        chunksize=1,
        desc="Fitting spectra"
    )

    fitted_values = (
        pd.concat(rows, ignore_index=True)  # bring everything into one DataFrame
    )

    fitted_values = fitted_values.round(5)
    fitted_values.to_csv(f"fitted_synthetic_just_elements.csv", index=False)

