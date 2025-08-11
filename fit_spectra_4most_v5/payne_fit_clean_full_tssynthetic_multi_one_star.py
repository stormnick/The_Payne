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

def fit_one_spectrum(file, stellar_rv, folder, payne_parameters, snr=100):
    print(f"Fitting {file}")
    wavelength_obs, flux_obs = np.loadtxt(f"{folder}/{file}", usecols=(0, 1), unpack=True, dtype=float)

    if stellar_rv != 0:
        wavelength_obs = wavelength_obs / (1 + (stellar_rv / 299792.458))

    stellar_parameters = create_default_stellar_parameters(payne_parameters)

    h_line_cores = pd.read_csv("../linemasks/h_cores.csv")
    h_line_cores = list(h_line_cores['ll'])

    wavelength_obs, flux_obs = process_spectra(payne_parameters.wavelength_payne, wavelength_obs, flux_obs, h_line_cores,
                                               h_line_core_mask_dlam=0.2)

    stellar_parameters = fit_stellar_parameters(stellar_parameters, payne_parameters, wavelength_obs, flux_obs,
                                                silent=True, sigma_flux=1/snr)

    labels = payne_parameters.labels

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
                                                 flux_obs, silent=True, sigma_flux=1/snr)


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


    plot_fitted_payne(wavelength_payne, final_parameters, payne_parameters.payne_coeffs, wavelength_obs, flux_obs, payne_parameters.labels, None)#, real_labels2=[4.665, 1.32, -2.657, 1.58])

    return new_row_df

def _wrapper(path, folder, payne_parameters, snr=100):
    stellar_rv = 0
    return fit_one_spectrum(path, stellar_rv, folder, payne_parameters, snr)

if __name__ == '__main__':
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_batch01_medium_test2training_reducedlogg_altarch_2025-06-16-06-28-26.npz"
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
    folder = "/Users/storm/PycharmProjects/payne/observed_spectra_to_test/benchmark/"
    folder = "/Users/storm/PhD_2025/02.22 Payne/real_spectra_to_fit/converted/"
    folder = "/Users/storm/PhD_2022-2025/Spectra/some_lowfeh_benchmark/norm_20k_degraded/"
    files = os.listdir(folder)

    # remove ".DS_Store"
    files.remove(".DS_Store")

    # remove any starting with "j" or "t"
    files = [f for f in files if not f.startswith(("j", "t"))]

    start_fit_time = perf_counter()
    print(f"Fitting {len(files)} spectra... starting at {time.strftime('%H:%M:%S', time.localtime(start_fit_time))}")

    payne_parameters = PayneParams(
        payne_coeffs=payne_coeffs,
        wavelength_payne=wavelength_payne,
        labels=labels,
        x_min=x_min,
        x_max=x_max,
        resolution_val=resolution_val
    )

    file_to_fit = "UVES_alfTau"
    file_to_fit = "HD133442"
    file_to_fit = "hd140283_UVES_4most.txt"
    file_to_fit = "hd122563_UVES_4most.txt"
    file_to_fit = "hd84937_UVES_4most.txt"

    for file in files:
        if file_to_fit in file:
            rows = _wrapper(file, folder, payne_parameters, 10)


    end_fit_time = perf_counter()
    print(f"Fitting completed in {(end_fit_time - start_fit_time) / 60:.2f} min.")
