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
from payne_fit_clean_full import (fit_teff, fit_logg, fit_feh, fit_one_xfe_element, process_spectra, load_payne,
                                  fit_teff_logg, plot_fitted_payne)
import random
from dask.distributed import Client, LocalCluster, wait
import dask.dataframe as dd
from dask import delayed
import os
from tqdm import tqdm

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 03.03.25

def fit_one_spectrum(file, stellar_rv, true_parameter):
    snr_to_do = int(true_parameter['snr'])
    #print(f"Fitting {file}")
    continuum = np.load(f"{file.replace(f'snr{snr_to_do}.0', f'cont')}")
    spectra = np.load(f"{file}")
    wavelength_obs = continuum[0]
    flux_obs = spectra[1] / continuum[1]
    start_time = time.perf_counter()
    h_line_cores = pd.read_csv("../linemasks/h_cores.csv")
    h_line_cores = list(h_line_cores['ll'])
    wavelength_obs, flux_obs = process_spectra(wavelength_payne, wavelength_obs, flux_obs, h_line_cores,
                                               h_line_core_mask_dlam=0.2)
    final_parameters = {}
    final_parameters_std = {}

    final_parameters["teff"] = true_parameter['teff'] / 1000
    final_parameters_std["teff"] = -1
    final_parameters["logg"] = true_parameter['logg']
    final_parameters["doppler_shift"] = 0
    final_parameters_std["logg"] = -1
    final_parameters_std["doppler_shift"] = -1

    final_parameters["feh"] = true_parameter['feh']
    final_parameters["vmic"] = true_parameter['vmic']
    final_parameters["vsini"] = true_parameter['vsini']
    final_parameters["vmac"] = 0
    final_parameters_std["feh"] = -1
    final_parameters_std["vmic"] = -1
    final_parameters_std["vsini"] = -1
    final_parameters_std["vmac"] = -1
    # 4. REMAINING ELEMENTS ONE-BY-ONE
    # find how many _Fe labels are there
    elements_to_fit = []
    for i, label in enumerate(labels):
        if label.endswith("_Fe"):
            elements_to_fit.append(label)
    # first we want to fit C_Fe, then O_Fe, then the rest
    priority = {"C_Fe": 0, "O_Fe": 1}  # everything else gets 2 by default
    elements_to_fit = sorted(
        (lbl for lbl in labels if lbl.endswith("_Fe")),
        key=lambda x: (priority.get(x, 2), x)  # (first by priority, then alphabetically)
    )

    final_parameters_input = final_parameters.copy()
    for element in elements_to_fit:
        final_parameters_input[element] = true_parameter[element]

    element_to_fit = "Mg_Fe"
    xfe, xfe_std = fit_one_xfe_element(final_parameters_input, element_to_fit, labels, payne_coeffs, x_min, x_max,
                                       stellar_rv, wavelength_obs, flux_obs,
                                       wavelength_payne, resolution_val, silent=True)

    final_parameters[element_to_fit] = xfe
    final_parameters_std[element_to_fit] = xfe_std
    #print(f"Fitted {file} in {time.perf_counter() - start_time:.2f} seconds")
    # PRINT RESULTS
    #for label in label_names:
    #    value = final_parameters[label]
    #    std_error = final_parameters_std[label]
    #    if label != 'teff':
    #        print(f"{label:<15}: {value:>10.3f} +/- {std_error:>10.3f}")
    #    else:
    #        print(f"{label:<15}: {value * 1000:>10.3f} +/- {std_error * 1000:>10.3f}")
    #if resolution_val is not None:
    #    print(f"{'Resolution':<15}: {int(resolution_val):>10}")
    if "/" in file:
        filename_to_save = file.split("/")[-1]
    else:
        filename_to_save = file
    # add to fitted_values
    # --- rename the “std” keys -----------------------------------------------
    final_parameters_std_renamed = {f"{k}_std": v for k, v in final_parameters_std.items()}

    # --- merge the two dicts ---------------------------------------------------
    row_dict = {
        "spectraname": filename_to_save,
        "snr": snr_to_do,
        **final_parameters,  # original values
        **final_parameters_std_renamed  # same keys, but with “_std” suffix
    }

    # --- build the 1-row DataFrame --------------------------------------------
    new_row_df = pd.DataFrame([row_dict])

    return new_row_df

def fit_one_spectrum_delayed(path, stellar_rv, true_parameter):
    """
    Wraps your existing routine so it can run on a Dask worker.

    It *must* return a 1-row pandas.DataFrame (or Series) with the
    exact columns defined above.
    """
    return fit_one_spectrum(path, stellar_rv, true_parameter)  # ← your original code


if __name__ == '__main__':
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_2025-03-27-08-06-34.npz"

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
    snrs_to_do = [10, 20, 50, 100, 250, 1000]
    #folder_spectra = f"/Users/storm/PycharmProjects/payne/ts_nlte_grid_july2024/snr{int(snr_to_do)}/"
    folder_spectra = f"/Users/storm/PhD_2025/spectra_4most/batch0_nlte_4mostified_v3/"
    # only take those with "hrs" in it
    # grab every filename ending in ".npy"
    all_files = [f for f in os.listdir(folder_spectra) if f.endswith(".npy")]

    ids = sorted({f.split('.')[0] for f in all_files})

    random.seed(42)
    chosen_ids = set(random.sample(ids, 10000))

    true_values = pd.read_csv("/Users/storm/PycharmProjects/payne/ts_nlte_grid_apr2024/spectra_parameters_nlte_batch0_v3.csv")
    true_values["spectraname"] = true_values["specname"]

    def get_spectra(snr_to_do: int):
        tag = f"hrs_snr{snr_to_do}.0"
        return [
            os.path.join(folder_spectra, f)
            for f in all_files
            if tag in f and f.split('.')[0] in chosen_ids
        ]

    all_snr_files = []
    for snr_to_do in snrs_to_do:
        files = get_spectra(snr_to_do)
        all_snr_files.extend(files)
    print(f"Fitting {len(all_snr_files)} spectra with SNR {snrs_to_do}")

    true_parameters = []

    for file in all_snr_files:
        # find that row in true_values["spectraname"]
        if "/" in file:
            filename_to_save = file.split("/")[-1]
        else:
            filename_to_save = file
        # find snr_to_do
        snr_to_do = filename_to_save.split('.0')[0].split("_hrs_snr")[1]
        row_df = true_values[true_values["spectraname"] == filename_to_save.replace(f"_hrs_snr{snr_to_do}.0.npy", "")]
        true_parameters.append(row_df.to_dict(orient="records")[0])
        true_parameters[-1]["snr"] = snr_to_do

    ddf = pd.DataFrame()  # tells Dask the schema

    # --- 3. build the task graph -------------------------------------------------
    for f, true_parameter in tqdm(zip(all_snr_files, true_parameters), total=len(all_snr_files)):
        new_row_df = fit_one_spectrum_delayed(f, stellar_rv, true_parameter)
        ddf = pd.concat([ddf, new_row_df], ignore_index=True)

    ddf = ddf.round(5)
    ddf.to_csv(f"just_elements_fitted_mg_varied_snr.csv", index=False)

