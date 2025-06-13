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
from tqdm.contrib.concurrent import process_map
from itertools import repeat
from time import perf_counter

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 03.03.25

def fit_one_spectrum(file, stellar_rv, folder, payne_coeffs, wavelength_payne, labels, label_names, resolution_val=None):
    x_min = list(payne_coeffs[-2])
    x_max = list(payne_coeffs[-1])
    print(f"Fitting {file}")

    wavelength_obs, flux_obs = np.loadtxt(f"{folder}/{file}", usecols=(0,1), unpack=True, dtype=float)
    start_time = time.perf_counter()
    h_line_cores = pd.read_csv("../linemasks/h_cores.csv")
    h_line_cores = list(h_line_cores['ll'])
    wavelength_obs, flux_obs = process_spectra(wavelength_payne, wavelength_obs, flux_obs, h_line_cores,
                                               h_line_core_mask_dlam=0.2)
    final_parameters = {}
    final_parameters_std = {}
    # 1. TEFF
    # fits teff, logg, feh, vmac, rv for h-alpha lines
    # teff, teff_std = fit_teff(labels, payne_coeffs, x_min, x_max, stellar_rv, h_line_cores, wavelength_obs,
    #                          flux_obs, wavelength_payne, resolution_val, silent=True)
    # final_parameters["teff"] = teff
    # final_parameters_std["teff"] = teff_std
    # 2. LOGG, FEH, VMAC, RV, also fit Mg, Ca
    teff, teff_std, logg, logg_std, doppler_shift, doppler_shift_std, popt = fit_teff_logg(labels, payne_coeffs, x_min,
                                                                                     x_max, stellar_rv,
                                                                                     wavelength_obs, flux_obs,
                                                                                     wavelength_payne,
                                                                                     resolution_val, silent=False)

    if logg > 30.5:
        teff, teff_std, logg, logg_std, doppler_shift, doppler_shift_std, _ = fit_teff_logg(labels, payne_coeffs, x_min,
                                                                                         x_max, stellar_rv,
                                                                                         wavelength_obs, flux_obs,
                                                                                         wavelength_payne,
                                                                                         resolution_val, silent=False, do_hydrogen_lines=True, p0_input=popt)

    final_parameters["teff"] = teff
    final_parameters_std["teff"] = teff_std
    final_parameters["logg"] = logg
    final_parameters["doppler_shift"] = doppler_shift
    final_parameters_std["logg"] = logg_std
    final_parameters_std["doppler_shift"] = doppler_shift_std

    # 3. FEH, VMIC, VMAC
    feh, feh_std, vmic, vmic_std, vsini, vsini_std, vmac, vmac_std = fit_feh(final_parameters, labels, payne_coeffs,
                                                                             x_min, x_max, stellar_rv,
                                                                             wavelength_obs,
                                                                             flux_obs, wavelength_payne,
                                                                             resolution_val,
                                                                             silent=False, fit_vsini=True,
                                                                             fit_vmac=False)
    final_parameters["feh"] = feh
    final_parameters["vmic"] = vmic
    final_parameters["vsini"] = vsini
    final_parameters["vmac"] = vmac
    final_parameters_std["feh"] = feh_std
    final_parameters_std["vmic"] = vmic_std
    final_parameters_std["vsini"] = vsini_std
    final_parameters_std["vmac"] = vmac_std
    # 4. REMAINING ELEMENTS ONE-BY-ONE
    # first we want to fit C_Fe, then O_Fe, then the rest
    priority = {"C_Fe": 0, "O_Fe": 1}  # everything else gets 2 by default
    elements_to_fit = sorted(
        (lbl for lbl in labels if lbl.endswith("_Fe")),
        key=lambda x: (priority.get(x, 2), x)  # (first by priority, then alphabetically)
    )
    elements_to_fit.append("A_Li")  # add A_Li to the end of the list
    for element_to_fit in elements_to_fit:
        xfe, xfe_std = fit_one_xfe_element(final_parameters, element_to_fit, labels, payne_coeffs, x_min, x_max,
                                           stellar_rv, wavelength_obs, flux_obs,
                                           wavelength_payne, resolution_val, silent=True)

        if xfe < -90:
            index = labels.index(element_to_fit)
            final_parameters[element_to_fit] = x_min[index]
            final_parameters_std[element_to_fit] = -99
        else:
            final_parameters[element_to_fit] = xfe
            final_parameters_std[element_to_fit] = xfe_std
    print(f"Fitted {file} in {time.perf_counter() - start_time:.2f} seconds")
    # PRINT RESULTS
    for label in label_names:
        value = final_parameters[label]
        std_error = final_parameters_std[label]
        if label != 'teff':
            print(f"{label:<15}: {value:>10.3f} +/- {std_error:>10.3f}")
        else:
            print(f"{label:<15}: {value * 1000:>10.3f} +/- {std_error * 1000:>10.3f}")
    if resolution_val is not None:
        print(f"{'Resolution':<15}: {int(resolution_val):>10}")
    if "/" in file:
        filename_to_save = file.split("/")[-1]
    else:
        filename_to_save = file

    plot_fitted_payne(wavelength_payne, final_parameters, payne_coeffs, wavelength_obs, flux_obs, labels, None, real_labels2=[4.665, 1.32, -2.657, 1.58])

    # add to fitted_values
    new_row_df = pd.DataFrame(
        [[filename_to_save, *final_parameters.values(), *final_parameters_std.values()]],
        columns=["spectraname"] + list(final_parameters.keys()) + [f"{name}_std" for name in final_parameters.keys()]
    )

    return new_row_df

def _wrapper(path, folder, payne_coeffs, wavelength_payne, labels, label_names, resolution_val):
    stellar_rv = 0
    return fit_one_spectrum(path, stellar_rv, folder, payne_coeffs, wavelength_payne, labels, label_names, resolution_val)


if __name__ == '__main__':
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_batch01_2025-06-09-10-53-13.npz"
    #path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_bigpayne_2025-06-09-16-42-57.npz"
    #path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_2025-06-06-14-18-21.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_batch01_medium_test2training_2025-06-10-10-09-41.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_batch01_medium_test2training_reducedlogg_2025-06-10-14-52-32.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_batch01_medium_test2training_reducedlogg_2025-06-11-08-02-05.npz"
    #path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_batch01_medium_test2training_reducedlogg_altarch_2025-06-12-10-27-38.npz"

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
    files = os.listdir(folder)

    # remove ".DS_Store"
    files.remove(".DS_Store")

    start_fit_time = perf_counter()
    print(f"Fitting {len(files)} spectra... starting at {time.strftime('%H:%M:%S', time.localtime())}")

    file_to_fit = "NARVAL_HD84937.txt"

    for file in files:
        if file_to_fit in file:
            rows = _wrapper(file, folder, payne_coeffs, wavelength_payne, labels, label_names, resolution_val)


    end_fit_time = perf_counter()
    print(f"Fitting completed in {(end_fit_time - start_fit_time) / 60:.2f} min.")