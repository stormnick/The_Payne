from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy
from The_Payne import utils
from The_Payne import spectral_model
from The_Payne import fitting
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
        labels = tmp["label_names"]
    except KeyError:
        lmin, lmax = 5330, 5615
        wavelength = np.linspace(lmin, lmax + 0.001, 28501)
        labels = ['teff', 'logg', 'feh', 'vmic', 'C_Fe', 'Mg_Fe', 'Ca_Fe', 'Ti_Fe', 'Ba_Fe']
    tmp.close()
    payne_coeffs = (w_array_0, w_array_1, w_array_2,
                    b_array_0, b_array_1, b_array_2,
                    x_min, x_max)
    return payne_coeffs, wavelength, labels


def make_model_spectrum_for_curve_fit(payne_coeffs, wavelength_payne, resolution_val=None):
    def model_spectrum_for_curve_fit(wavelength_obs, *params_to_fit):
        real_labels = np.array(params_to_fit[:-3])
        vrot = params_to_fit[-3]
        vmac = params_to_fit[-2]
        doppler_shift = params_to_fit[-1]

        scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
        spec_payne = spectral_model.get_spectrum_from_neural_net(
            scaled_labels=scaled_labels,
            NN_coeffs=payne_coeffs,
            kovalev_alt=True
        )

        wavelength_payne_ = wavelength_payne

        if resolution_val is not None:
            wavelength_payne_, spec_payne = conv_res(wavelength_payne_, spec_payne, resolution_val)
        #if vrot > 0:
            #wavelength_payne_, spec_payne = conv_rotation(wavelength_payne_, spec_payne, vrot)
        if vmac > 0:
            wavelength_payne_, spec_payne = conv_macroturbulence(wavelength_payne_, spec_payne, vmac)

        f_interp = interp1d(
            wavelength_payne_,
            spec_payne,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

        wavelength_obs = wavelength_obs / (1 + (doppler_shift / 299792.))

        return f_interp(wavelength_obs)

    return model_spectrum_for_curve_fit


if __name__ == '__main__':
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_alt_smallerldelta_ts_nlte_lesselements_hr10_2025-02-27-08-43-08.npz"
    payne_coeffs, wavelength_payne, labels = load_payne(path_model)

    real_labels = [5.777, 4.44, 0.0, 1.0, 0., 0., 0., 0, 0]
    scaled_labels = (
        (np.array(real_labels) - payne_coeffs[-2]) /
        (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
    )
    wavelength_obs = wavelength_payne
    flux_obs = spectral_model.get_spectrum_from_neural_net(
        scaled_labels=scaled_labels,
        NN_coeffs=payne_coeffs,
        kovalev_alt=True
    )

    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/KPNO_FTS_flux_2960_13000_Kurucz1984.txt", dtype=float, unpack=True)
    wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PycharmProjects/4most/Victor/spectra_victor_jan25/G48-29", dtype=float, unpack=True)
    resolution_val = 20_000

    l_cut = (wavelength_obs > wavelength_payne[0]) & (wavelength_obs < wavelength_payne[-1])
    wavelength_obs = wavelength_obs[l_cut]
    flux_obs = flux_obs[l_cut]

    p0 = [4.777, 4.44, -2.0, 1.5, -1., -1., -1., -1, -1, 1, 1, 1]

    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne,
        resolution_val=resolution_val
    )

    popt, pcov = curve_fit(
        model_func,
        wavelength_obs,
        flux_obs,
        p0=p0,
        bounds=([3.5, 0, -4, 0.5, -3, -3, -3, -3, -3, 0, 0, -10], [8, 5, 0.5, 3, 3, 3, 3, 3, 3, 15, 15, 10])
    )

    print(popt)
    labels.append('vrot')
    labels.append('vmac')
    labels.append('doppler_shift')
    for label, value in zip(labels, popt):
        if label != 'teff':
            print(f"{label:<15}: {value:>10.3f}")
        else:
            print(f"{label:<15}: {value*1000:>10.3f}")

    doppler_shift = popt[-1]
    vmac = popt[-2]
    vrot = popt[-3]

    real_labels = popt[:-3]
    scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
    payne_fitted_spectra = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels,
                                                                  NN_coeffs=payne_coeffs, kovalev_alt=True)

    wavelength_payne_plot = wavelength_payne
    if resolution_val is not None:
        wavelength_payne_plot, payne_fitted_spectra = conv_res(wavelength_payne_plot, payne_fitted_spectra, resolution_val)
    # if vrot > 0:
    # wavelength_payne_plot, payne_fitted_spectra = conv_rotation(wavelength_payne_plot, payne_fitted_spectra, vrot)
    if vmac > 0:
        wavelength_payne_plot, payne_fitted_spectra = conv_macroturbulence(wavelength_payne_plot, payne_fitted_spectra, vmac)

    plt.figure(figsize=(18, 6))
    plt.scatter(wavelength_obs / (1 + (doppler_shift / 299792.)), flux_obs, label="Observed", s=3, color='k')
    plt.plot(wavelength_payne_plot, payne_fitted_spectra, label="Payne", color='r')
    plt.show()