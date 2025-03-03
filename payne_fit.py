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

def payne_fit():
    real_labels = [5.777, 4.44, 0.0, 1.0, 0., 0., 0., 0, 0]

    scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
    spec_payne = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels,
                                                                  NN_coeffs=payne_coeffs, kovalev=True)


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

        if resolution_val is not None:
            _, spec_payne = conv_res(wavelength_payne, spec_payne, resolution_val)

        # _, spec_payne = conv_rotation(wavelength_payne, spec_payne, vrot)
        # _, spec_payne = conv_macroturbulence(wavelength_payne, spec_payne, vmac)

        f_interp = interp1d(
            wavelength_payne,
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

    wavelength    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_alt_smallerldelta_ts_nlte_lesselements_hr10_2025-02-27-08-43-08.npz"
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

    wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/KPNO_FTS_flux_2960_13000_Kurucz1984.txt", dtype=float, unpack=True)

    l_cut = (sun_wavelength > lmin) & (sun_wavelength < lmax)
    sun_wavelength = sun_wavelength[l_cut]
    sun_flux = sun_flux[l_cut]

    p0 = [4.777, 2.44, -2.0, 1.5, 0., 0., 0., 0, 0, 0, 0, 0]

    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne,
        resolution_val=None
    )


    popt, pcov = curve_fit(
        model_func,
        wavelength_obs,
        flux_obs,
        p0=p0
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
    # compare with real labels
    print(np.array(real_labels) - popt[:-3])


    #real_labels = [5.777, 4.44, 0.0, 1.0, 0., 0., 0., 0, 0]
    #scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
    #real_spec_4most = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels,
    #                                                              NN_coeffs=payne_coeffs, kovalev_alt=True)
    #plt.plot(wavelength_payne, real_spec_4most)
    #plt.show()_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/KPNO_FTS_flux_2960_13000_Kurucz1984.txt", dtype=float, unpack=True)

    l_cut = (sun_wavelength > lmin) & (sun_wavelength < lmax)
    sun_wavelength = sun_wavelength[l_cut]
    sun_flux = sun_flux[l_cut]

    p0 = [4.777, 2.44, -2.0, 1.5, 0., 0., 0., 0, 0, 0, 0, 0]

    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne,
        resolution_val=None
    )


    popt, pcov = curve_fit(
        model_func,
        wavelength_obs,
        flux_obs,
        p0=p0
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
    # compare with real labels
    print(np.array(real_labels) - popt[:-3])


    #real_labels = [5.777, 4.44, 0.0, 1.0, 0., 0., 0., 0, 0]
    #scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
    #real_spec_4most = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels,
    #                                                              NN_coeffs=payne_coeffs, kovalev_alt=True)
    #plt.plot(wavelength_payne, real_spec_4most)
    #plt.show()