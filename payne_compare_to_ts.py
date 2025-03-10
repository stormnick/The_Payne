from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy
from The_Payne import spectral_model
from convolve_vmac import *
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 03.03.25


def load_payne(path_model):
    tmp = np.load(path_model)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    try:
        wavelength = tmp["wavelength"]
        labels = list(tmp["label_names"])
    except KeyError:
        lmin, lmax = 5330, 5615
        wavelength = np.linspace(lmin, lmax + 0.001, 28501)
        labels = ['teff', 'logg', 'feh', 'vmic', 'C_Fe', 'Mg_Fe', 'Ca_Fe', 'Ti_Fe', 'Ba_Fe']
    tmp.close()
    payne_coeffs = (w_array_0, w_array_1, w_array_2,
                    b_array_0, b_array_1, b_array_2,
                    x_min, x_max)
    return payne_coeffs, wavelength, labels


def scale_back(x, x_min, x_max, label_name=None):
    x = np.array(x)
    x_min = np.array(x_min)
    x_max = np.array(x_max)
    return_value = (x + 0.5) * (x_max - x_min) + x_min
    if label_name == "teff":
        return_value = return_value * 1000
    return list(return_value)

if __name__ == '__main__':
    path_models = ["/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr3_2025-03-04-01-03-54.npz",
                   "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr10_2025-03-05-01-50-15.npz",
                   "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr13_2025-03-04-03-09-20.npz",
                   "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr15n_2025-03-04-19-35-44.npz"]

    labels_stellar = [5.777, 4.4, 0.0, 1.0]
    wavelength_payne_full = []
    flux_payne_full = []

    for path_model in path_models:
        payne_coeffs, wavelength_payne, labels = load_payne(path_model)
        labels_individual = labels_stellar + (len(labels) - 4) * [0]

        scaled_labels = (np.asarray(labels_individual) - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
        payne_spectra = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels,
                                                                        NN_coeffs=payne_coeffs, kovalev_alt=True)

        wavelength_payne_full.append(wavelength_payne)
        flux_payne_full.append(payne_spectra)

    # flatten the lists
    wavelength_payne_full = [item for sublist in wavelength_payne_full for item in sublist]
    flux_payne_full = [item for sublist in flux_payne_full for item in sublist]

    wavelength_ts, flux_ts = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/KPNO_FTS_flux_2960_13000_Kurucz1984.txt", dtype=float, unpack=True)

    l_cut = (wavelength_ts > np.min(wavelength_payne_full)) & (wavelength_ts < np.max(wavelength_payne_full))
    wavelength_ts = wavelength_ts[l_cut]
    flux_ts = flux_ts[l_cut]


    plt.figure(figsize=(18, 6))
    plt.scatter(wavelength_payne_full, flux_payne_full, label="Payne", s=3, color='k')
    plt.plot(wavelength_ts, flux_ts, label="TS", color='r')
    plt.ylim(0.0, 1.05)
    plt.xlim(np.min(wavelength_payne_full), np.max(wavelength_payne_full))
    plt.legend()
    plt.show()