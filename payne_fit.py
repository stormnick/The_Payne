from __future__ import annotations

import corner
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

        if vmac > 1e-2:
            wavelength_payne_, spec_payne = conv_macroturbulence(wavelength_payne_, spec_payne, vmac)
        if vrot > 1e-2:
            wavelength_payne_, spec_payne = conv_rotation(wavelength_payne_, spec_payne, vrot)
        if resolution_val is not None:
            wavelength_payne_, spec_payne = conv_res(wavelength_payne_, spec_payne, resolution_val)

        wavelength_payne_ = wavelength_payne_ * (1 + (doppler_shift / 299792.))

        f_interp = interp1d(
            wavelength_payne_,
            spec_payne,
            kind='linear',
            bounds_error=False,
            fill_value=1
        )

        interpolated_spectrum = f_interp(wavelength_obs)

        #plt.scatter(wavelength_obs, flux_obs, s=3, color='k')
        #plt.plot(wavelength_obs, interpolated_spectrum, color='r')
        #plt.show()

        # calculate chi-squared
        #chi_squared = np.sum((interpolated_spectrum - flux_obs) ** 2)
        #print(chi_squared)

        return interpolated_spectrum

    return model_spectrum_for_curve_fit

def scale_back(x, x_min, x_max, label_name=None):
    x = np.array(x)
    x_min = np.array(x_min)
    x_max = np.array(x_max)
    return_value = (x + 0.5) * (x_max - x_min) + x_min
    if label_name == "teff":
        return_value = return_value * 1000
    return list(return_value)

if __name__ == '__main__':
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_alt_smallerldelta_ts_nlte_lesselements_hr10_2025-02-27-08-43-08.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr3_2025-03-04-01-03-54.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr10_2025-03-05-01-50-15.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr13_2025-03-04-03-09-20.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr15n_2025-03-12-08-32-42.npz"
    payne_coeffs, wavelength_payne, labels = load_payne(path_model)
    x_min = list(payne_coeffs[-2])
    x_max = list(payne_coeffs[-1])

    wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/KPNO_FTS_flux_2960_13000_Kurucz1984.txt", dtype=float, unpack=True)
    wavelength_obs, flux_obs = np.loadtxt("./ts_spectra/sun_nlte.spec", dtype=float, unpack=True, usecols=(0, 1))
    wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PycharmProjects/4most/Victor/spectra_victor_jan25/G48-29", dtype=float, unpack=True)
    wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PycharmProjects/4most/Victor/spectra_victor_jan25/G64-12", dtype=float, unpack=True)
    #data = np.loadtxt("18Sco_cont_norm.txt", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = data[:, 0], data[:, 1]
    #wavelength_obs, flux_obs = np.loadtxt("ADP_18sco_snr396_HARPS_17.707g_2.norm", dtype=float, unpack=True, usecols=(0, 2), skiprows=1)
    #wavelength_obs, flux_obs = np.loadtxt("./ts_spectra/synthetic_data_sun_nlte_full.txt", dtype=float, unpack=True, usecols=(0, 1))

    mask = (flux_obs > 0.0) & (flux_obs < 1.2)
    wavelength_obs = wavelength_obs[mask]
    flux_obs = flux_obs[mask]

    resolution_val = None

    l_cut = (wavelength_obs > wavelength_payne[0]) & (wavelength_obs < wavelength_payne[-1])
    wavelength_obs = wavelength_obs[l_cut]
    flux_obs = flux_obs[l_cut]

    #p0 = [7.777, 2.94, 0.0, 1.5, -2., -2., -2., -2, 0, 3, 0]
    p0 = [6.777, 4.54, 0.0, 1.0] + (len(labels) - 4) * [0] + [3,3, 1]

    #p0 = scale_back([0] * (len(p0) - 3), payne_coeffs[-2], payne_coeffs[-1], label_name=None)
    # add extra 3 0s
    #p0 += [0, 0, 0]

    #def_bounds = ([3.5, 0, -4, 0.5, -3, -3, -3, -3, 0, 0, -20], [8, 5, 0.5, 3, 3, 3, 3, 3, 1e-5, 15, 20])
    def_bounds = (x_min + [0, 0, -1], x_max + [10, 10, 1])

    input_values = [None] * len(p0)
    #input_values = (6.394, 4.4297, -2.8919, 1.4783, None, None, None, None, None, 3.7822, 0, 0)
    #input_values = (4287.7906, 4.5535, -0.5972, 0.6142, None, None, None, None, None, 5.399, 0, 0)
    #input_values = (6290.449, 4.6668, -3.7677, 1.1195, None, None, None, None, None, 1.2229, 0, 0)
    #input_values[0:3] = (5777, 4.44)
    for i, input_value in enumerate(input_values):
        if input_value is not None:
            if i == 0 and input_value > 100:
                input_value /= 1000
            p0[i] = input_value
            def_bounds[0][i] = input_value - 1e-3
            def_bounds[1][i] = input_value + 1e-3

    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne,
        resolution_val=resolution_val
    )

    print("Fitting...")

    popt, pcov = curve_fit(
        model_func,
        wavelength_obs,
        flux_obs,
        p0=p0,
        bounds=def_bounds,
    )

    print("Done fitting.")

    labels.append('vrot')
    labels.append('vmac')
    labels.append('doppler_shift')

    num_samples = 10000
    samples = np.random.multivariate_normal(mean=popt, cov=pcov, size=num_samples)

    # 4. Make the corner plot
    # --------------------------------------------------
    figure = corner.corner(
        samples,
        labels=labels,  # Replace with names of your parameters
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 12}
    )

    plt.show()

    #print(popt)
    for label, value in zip(labels, popt):
        if label != 'teff':
            print(f"{label:<15}: {value:>10.3f} +/- {np.sqrt(np.diag(pcov))[labels.index(label)]:>10.3f}")
        else:
            print(f"{label:<15}: {value*1000:>10.3f} +/- {np.sqrt(np.diag(pcov))[labels.index(label)]*1000:>10.3f}")
    if resolution_val is not None:
        print(f"{'Resolution':<15}: {int(resolution_val):>10}")

    doppler_shift = popt[-1]
    vmac = popt[-2]
    vrot = popt[-3]

    real_labels = popt[:-3]
    scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
    payne_fitted_spectra = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels,
                                                                  NN_coeffs=payne_coeffs, kovalev_alt=True)

    wavelength_payne_plot = wavelength_payne
    if vmac > 1e-3:
        wavelength_payne_plot, payne_fitted_spectra = conv_macroturbulence(wavelength_payne_plot, payne_fitted_spectra, vmac)
    if vrot > 1e-3:
        wavelength_payne_plot, payne_fitted_spectra = conv_rotation(wavelength_payne_plot, payne_fitted_spectra, vrot)
    if resolution_val is not None:
        wavelength_payne_plot, payne_fitted_spectra = conv_res(wavelength_payne_plot, payne_fitted_spectra, resolution_val)

    plt.figure(figsize=(18, 6))
    plt.scatter(wavelength_obs, flux_obs, label="Observed", s=3, color='k')
    plt.plot(wavelength_payne_plot * (1 + (doppler_shift / 299792.)), payne_fitted_spectra, label="Payne", color='r')
    #plt.plot(wavelength_test * (1 + (doppler_shift / 299792.)), flux_test, label="Payne test", color='b')
    plt.ylim(0.0, 1.05)
    plt.xlim(wavelength_payne_plot[0], wavelength_payne_plot[-1])
    plt.show()