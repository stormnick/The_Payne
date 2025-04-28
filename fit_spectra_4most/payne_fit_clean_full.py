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

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 03.03.25

def apply_on_segments(
        wavelength,            # 1-D array-like
        spectrum,              # 1-D array-like, same length
        func,                  # callable(wl_seg, sp_seg, *a, **kw) → (wl_out, sp_out)
        *func_args,            # forwarded positional args (e.g. vrot)
        spacing_tolerance=2.0, # gap = diff > tol × median_spacing
        assume_sorted=True,    # set False if the array is unsorted
        **func_kwargs):        # forwarded keyword args
    """
    Call `func` independently on each uniformly-spaced wavelength segment.

    Returns
    -------
    wavelength_out, spectrum_out : np.ndarray
        Concatenated outputs in the original order.
    """
    wl = np.asarray(wavelength)
    sp = np.asarray(spectrum)

    if wl.ndim != 1 or wl.shape != sp.shape:
        raise ValueError("`wavelength` and `spectrum` must be 1-D and equally long")

    # sort once if needed (keeps wavelengths increasing)
    if not assume_sorted:
        order = np.argsort(wl)
        wl, sp = wl[order], sp[order]

    # identify “large” gaps
    diffs           = np.diff(wl)
    median_spacing  = np.median(diffs)
    gap_locations   = np.where(diffs > spacing_tolerance * median_spacing)[0]

    # segment boundaries (inclusive start, exclusive end)
    bounds = np.concatenate(([0], gap_locations + 1, [len(wl)]))

    w_out, s_out = [], []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        wl_seg, sp_seg = wl[start:end], sp[start:end]
        wl_conv, sp_conv = func(wl_seg, sp_seg, *func_args, **func_kwargs)
        w_out.append(np.asarray(wl_conv))
        s_out.append(np.asarray(sp_conv))

    return np.concatenate(w_out), np.concatenate(s_out)

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


def make_model_spectrum_for_curve_fit(payne_coeffs, wavelength_payne, input_values, resolution_val=None,
            pixel_limits=None):
    def model_spectrum_for_curve_fit(wavelength_obs, *params_to_fit):
        spectra_params = np.array(input_values).copy().astype(float)
        j = 0
        for i, input_value in enumerate(input_values):
            if input_value is None:
                spectra_params[i] = params_to_fit[j]
                j += 1

        vrot = spectra_params[-3]
        vmac = spectra_params[-2]
        doppler_shift = spectra_params[-1]

        real_labels = spectra_params[:-3]

        scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
        spec_payne = spectral_model.get_spectrum_from_neural_net(
            scaled_labels=scaled_labels,
            NN_coeffs=payne_coeffs,
            kovalev_alt=True,
            pixel_limits=pixel_limits
        )

        wavelength_payne_ = wavelength_payne

        if vmac > 0:
            wavelength_payne_, spec_payne = apply_on_segments(
                wavelength_payne_,
                spec_payne,
                conv_macroturbulence,       # <-- your original routine
                vmac                 # forwarded to conv_rotation
            )
        if vrot > 0:
            wavelength_payne_, spec_payne = apply_on_segments(
                wavelength_payne_,
                spec_payne,
                conv_rotation,       # <-- your original routine
                vrot                 # forwarded to conv_rotation
            )
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


def get_bounds_and_p0(p0, input_values):
    columns_to_pop = []
    for i, input_value in enumerate(input_values):
        if input_value is not None:
            if i == 0 and input_value > 100:
                input_value /= 1000
            p0[i] = input_value
            # remove that column
            columns_to_pop.append(i)
    # remove the columns from p0 and def_bounds
    for i in sorted(columns_to_pop, reverse=True):
        p0.pop(i)
        def_bounds[0].pop(i)
        def_bounds[1].pop(i)

    labels_to_fit = [True] * (len(labels) + 3)
    for i, input_value in enumerate(input_values):
        if input_value is not None:
            labels_to_fit[i] = False

    return p0, columns_to_pop, labels_to_fit


def cut_to_just_lines(wavelength_obs, flux_obs, fe_lines_to_use, obs_cut_aa=0.5, payne_cut_aa=0.75):
    # cut so that we only take the lines we want
    masks = []
    masks_payne = []
    for line in fe_lines_to_use:
        mask_one = (wavelength_obs > line - obs_cut_aa) & (wavelength_obs < line + obs_cut_aa)
        masks.append(mask_one)

        mask_payne = (wavelength_payne > line - payne_cut_aa) & (wavelength_payne < line + payne_cut_aa)
        masks_payne.append(mask_payne)
    # apply masks
    combined_mask = np.array(masks).any(axis=0)
    combined_mask_payne_ = np.array(masks_payne).any(axis=0)
    wavelength_obs_cut_to_lines_ = wavelength_obs[combined_mask]
    flux_obs_cut_to_lines_ = flux_obs[combined_mask]

    # combined_mask_payne_ = None

    if combined_mask_payne_ is not None:
        wavelength_payne_cut_ = wavelength_payne[combined_mask_payne_]
    else:
        wavelength_payne_cut_ = wavelength_payne

    return wavelength_obs_cut_to_lines_, flux_obs_cut_to_lines_, wavelength_payne_cut_, combined_mask_payne_

def get_default_p0_guess(labels, payne_coeffs, x_min, x_max, stellar_rv):
    p0 = scale_back([0] * (len(labels)), payne_coeffs[-2], payne_coeffs[-1], label_name=None)
    # add extra 3 0s
    p0 += [3, 3, 0]

    input_values = [None] * len(p0)
    def_bounds = (x_min + [0, 0, -10 + stellar_rv], x_max + [15, 15, 10 + stellar_rv])
    return p0, input_values, def_bounds

if __name__ == '__main__':
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_alt_smallerldelta_ts_nlte_lesselements_hr10_2025-02-27-08-43-08.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr3_2025-03-10-10-19-24.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr10_2025-03-12-07-46-13.npz"
    #path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr13_2025-03-12-08-29-38.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr15n_2025-03-12-08-32-42.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr15n_smalldelta_2025-03-24-14-04-40.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr15n_smalldelta_2025-03-24-14-46-44.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_2025-03-27-08-06-34.npz"
    #path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr10_2025-03-12-07-46-13.npz"
    """teff           :   5692.693 +/-      3.459
logg           :      4.469 +/-      0.013
feh            :     -0.261 +/-      0.004
vmic           :      1.385 +/-      0.014
A_Li           :      0.000 +/-      0.218
C_Fe           :      0.347 +/-      0.008
Ca_Fe          :      0.037 +/-      0.009
Ba_Fe          :     -0.058 +/-      0.021
vrot           :      0.000 +/-     -1.000
vmac           :      2.216 +/-      0.024
doppler_shift  :      0.327 +/-      0.008"""
    payne_coeffs, wavelength_payne, labels = load_payne(path_model)
    x_min = list(payne_coeffs[-2])
    x_max = list(payne_coeffs[-1])

    wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/KPNO_FTS_flux_2960_13000_Kurucz1984.txt", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/melchiors.txt", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/melchiors.txt", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/iag_solar_flux.txt", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = np.loadtxt("./ts_spectra/sun_nlte.spec", dtype=float, unpack=True, usecols=(0, 1))
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PycharmProjects/4most/Victor/spectra_victor_jan25/G48-29", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PycharmProjects/4most/Victor/spectra_victor_jan25/G64-12", dtype=float, unpack=True)
    #data = np.loadtxt("18Sco_cont_norm.txt", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = data[:, 0], data[:, 1]
    #wavelength_obs, flux_obs = np.loadtxt("ADP_18sco_snr396_HARPS_17.707g_2.norm", dtype=float, unpack=True, usecols=(0, 2), skiprows=1)
    #wavelength_obs, flux_obs = np.loadtxt("./ts_spectra/synthetic_data_sun_nlte_full.txt", dtype=float, unpack=True, usecols=(0, 1))
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/diff_stellar_spectra_MB/HARPS_HD122563.txt", dtype=float, unpack=True, usecols=(0, 1))
    stellar_rv = 0

    folder = "/Users/storm/Downloads/Cont/"
    folder_spectra = "/Users/storm/Downloads/Science/"
    file1 = "Sun_melchiors_spectrum.npy"
    continuum = np.load(f"{folder}{file1}")
    spectra = np.load(f"{folder_spectra}{file1}")
    wavelength_obs = continuum[0]
    flux_obs = spectra[1] / continuum[1]

    #wavelength_obs, flux_obs = conv_res(wavelength_obs, flux_obs, 20000)

    fe_lines = pd.read_csv("../fe_lines_hr_noblue.csv")
    fe_lines = pd.read_csv("../fe_lines_hr_good.csv")
    fe_lines_to_use = fe_lines["ll"]


    mask = (flux_obs > 0.0) & (flux_obs < 1.2)
    wavelength_obs = wavelength_obs[mask]
    flux_obs = flux_obs[mask]

    h_line_cores = [3970.072, 4101.734, 4340.462, 4861.323, 6562.696]
    h_line_core_mask_dlam = 0.5
    h_line_fit_mask = 15

    # mask H-line cores
    h_line_masks = []
    #for h_line_core in h_line_cores:
    #    h_

    h_alpha_mask = (wavelength_obs < 6562.8 - 0.5) | (wavelength_obs > 6562.8 + 0.5)
    wavelength_obs = wavelength_obs[h_alpha_mask]
    flux_obs = flux_obs[h_alpha_mask]

    resolution_val = None

    l_cut = (wavelength_obs > wavelength_payne[0]) & (wavelength_obs < wavelength_payne[-1])
    wavelength_obs = wavelength_obs[l_cut]
    flux_obs = flux_obs[l_cut]

    label_names = labels.copy()
    label_names.append('vrot')
    label_names.append('vmac')
    label_names.append('doppler_shift')

    final_parameters = {}
    final_parameters_std = {}

    # 1. TEFF
    # fits teff, logg, feh, vmac, rv for h-alpha lines
    teff_lines_to_use = h_line_cores

    p0, input_values, def_bounds = get_default_p0_guess(labels, payne_coeffs, x_min, x_max, stellar_rv)
    input_values = [None, None, None, 1] + [0] * (len(labels) - 4) + [None, 0, None]
    p0, columns_to_pop, labels_to_fit = get_bounds_and_p0(p0, input_values)

    #TODO: scaling of vmic with teff/logg/feh?

    wavelength_obs_cut_to_lines, flux_obs_cut_to_lines, wavelength_payne_cut, combined_mask_payne = cut_to_just_lines(wavelength_obs, flux_obs, teff_lines_to_use, obs_cut_aa=15, payne_cut_aa=20)

    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne_cut,
        input_values,
        resolution_val=resolution_val,
        pixel_limits=combined_mask_payne
    )

    print("Fitting...")

    time_start = time.perf_counter()

    popt, pcov = curve_fit(
        model_func,
        wavelength_obs_cut_to_lines,
        flux_obs_cut_to_lines,
        p0=p0,
        bounds=def_bounds,
    )

    print(f"Done fitting in {time.perf_counter() - time_start:.2f} seconds")

    # use fitted teff
    final_parameters["teff"] = popt[0]
    final_parameters_std["teff"] = np.sqrt(np.diag(pcov))[0]
    print(f"Fitted teff: {popt[0] * 1000:.3f} +/- {np.sqrt(np.diag(pcov))[0]:.3f}")

    # 2. LOGG, FEH, VMAC, RV, also fit Mg, Ca

    # load mg_fe and ca_fe lines
    mg_fe_lines = pd.read_csv("../linemasks/mg_triplet.csv")
    mg_fe_lines = mg_fe_lines['ll']
    ca_fe_lines = pd.read_csv("../linemasks/ca_triplet.csv")
    ca_fe_lines = ca_fe_lines['ll']

    # combine both
    logg_lines = list(mg_fe_lines) + list(ca_fe_lines)

    p0, input_values, def_bounds = get_default_p0_guess(labels, payne_coeffs, x_min, x_max, stellar_rv)
    input_values = [float(final_parameters["teff"]), None, None, 1] + [0] * (len(labels) - 4) + [None, 0, None]
    # find location of mg_fe and ca_fe in the labels
    mg_index = labels.index("Mg_Fe")
    ca_index = labels.index("Ca_Fe")

    # set mg and ca to 0
    input_values[mg_index] = None
    input_values[ca_index] = None
    p0, columns_to_pop, labels_to_fit = get_bounds_and_p0(p0, input_values)

    wavelength_obs_cut_to_lines, flux_obs_cut_to_lines, wavelength_payne_cut, combined_mask_payne = cut_to_just_lines(
        wavelength_obs, flux_obs, logg_lines, obs_cut_aa=5, payne_cut_aa=6)

    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne_cut,
        input_values,
        resolution_val=resolution_val,
        pixel_limits=combined_mask_payne
    )

    print("Fitting...")

    time_start = time.perf_counter()

    popt, pcov = curve_fit(
        model_func,
        wavelength_obs_cut_to_lines,
        flux_obs_cut_to_lines,
        p0=p0,
        bounds=def_bounds,
    )

    print(f"Done fitting in {time.perf_counter() - time_start:.2f} seconds")

    final_parameters["logg"] = popt[0]
    final_parameters["doppler_shift"] = popt[-1]
    final_parameters_std["logg"] = np.sqrt(np.diag(pcov))[0]
    final_parameters_std["doppler_shift"] = np.sqrt(np.diag(pcov))[-1]
    print(f"Fitted logg: {popt[0]:.3f} +/- {np.sqrt(np.diag(pcov))[0]:.3f}")
    print(f"Fitted doppler shift: {popt[-1]:.3f} +/- {np.sqrt(np.diag(pcov))[-1]:.3f}")

    # 3. FEH, VMIC, VMAC
    # load fe lines
    fe_lines = pd.read_csv("../fe_lines_hr_good.csv")
    fe_lines = list(fe_lines["ll"])
    p0, input_values, def_bounds = get_default_p0_guess(labels, payne_coeffs, x_min, x_max, stellar_rv)
    input_values = [float(final_parameters["teff"]), float(final_parameters["logg"]), None, None] + [0] * (len(labels) - 4) + [None, 0, float(final_parameters["doppler_shift"])]

    p0, columns_to_pop, labels_to_fit = get_bounds_and_p0(p0, input_values)

    wavelength_obs_cut_to_lines, flux_obs_cut_to_lines, wavelength_payne_cut, combined_mask_payne = cut_to_just_lines(
        wavelength_obs, flux_obs, fe_lines, obs_cut_aa=0.5, payne_cut_aa=0.75)

    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne_cut,
        input_values,
        resolution_val=resolution_val,
        pixel_limits=combined_mask_payne
    )

    print("Fitting...")

    time_start = time.perf_counter()

    popt, pcov = curve_fit(
        model_func,
        wavelength_obs_cut_to_lines,
        flux_obs_cut_to_lines,
        p0=p0,
        bounds=def_bounds,
    )

    print(f"Done fitting in {time.perf_counter() - time_start:.2f} seconds")

    final_parameters["feh"] = popt[0]
    final_parameters["vmic"] = popt[1]
    final_parameters["vsini"] = popt[-1]
    final_parameters_std["feh"] = np.sqrt(np.diag(pcov))[0]
    final_parameters_std["vmic"] = np.sqrt(np.diag(pcov))[1]
    final_parameters_std["vsini"] = np.sqrt(np.diag(pcov))[-1]
    print(f"Fitted feh: {popt[0]:.3f} +/- {np.sqrt(np.diag(pcov))[0]:.3f}")
    print(f"Fitted vmic: {popt[1]:.3f} +/- {np.sqrt(np.diag(pcov))[1]:.3f}")
    print(f"Fitted vsini: {popt[-1]:.3f} +/- {np.sqrt(np.diag(pcov))[-1]:.3f}")

    # 4. REMAINING ELEMENTS ONE-BY-ONE
    # find how many _Fe labels are there
    elements_to_fit = []
    for i, label in enumerate(labels):
        if label.endswith("_Fe"):
            elements_to_fit.append(label)

    for element_to_fit in elements_to_fit:
        path = f"../linemasks/{element_to_fit.split('_')[0].lower()}.csv"
        if os.path.exists(path):
            print("Loading linemask")
            element_lines = pd.read_csv(path)
            element_lines = list(element_lines['ll'])
            if element_to_fit != "C_Fe":
                dlam = 0.5
            else:
                dlam = 15
        else:
            element_lines = [5000]
            dlam = 2000

        p0, input_values, def_bounds = get_default_p0_guess(labels, payne_coeffs, x_min, x_max, stellar_rv)
        input_values = [float(final_parameters["teff"]), float(final_parameters["logg"]), float(final_parameters["feh"]),
                        float(final_parameters["vmic"])] + [0] * (len(labels) - 4) + [float(final_parameters["vsini"]), 0,
                                                                               float(final_parameters["doppler_shift"])]

        # TODO: take into account each other's abundances

        index_element = labels.index(element_to_fit)
        input_values[index_element] = None

        p0, columns_to_pop, labels_to_fit = get_bounds_and_p0(p0, input_values)

        print(input_values)

        wavelength_obs_cut_to_lines, flux_obs_cut_to_lines, wavelength_payne_cut, combined_mask_payne = cut_to_just_lines(
            wavelength_obs, flux_obs, element_lines, obs_cut_aa=dlam, payne_cut_aa=dlam * 1.5)

        model_func = make_model_spectrum_for_curve_fit(
            payne_coeffs,
            wavelength_payne_cut,
            input_values,
            resolution_val=resolution_val,
            pixel_limits=combined_mask_payne
        )

        print("Fitting...")

        time_start = time.perf_counter()

        popt, pcov = curve_fit(
            model_func,
            wavelength_obs_cut_to_lines,
            flux_obs_cut_to_lines,
            p0=p0,
            bounds=def_bounds,
        )

        print(f"Done fitting in {time.perf_counter() - time_start:.2f} seconds")

        final_parameters[element_to_fit] = float(popt[0])
        final_parameters_std[element_to_fit] = np.sqrt(np.diag(pcov))[0]

        print(f"Fitted {element_to_fit}: {popt[0]:.3f} +/- {np.sqrt(np.diag(pcov))[0]:.3f}")

        input_values[index_element] = float(popt[0])


    #final_params = np.array(input_values).copy().astype(float)
    #j = 0
    #for i, input_value in enumerate(input_values):
    #    if input_value is None:
    #        final_params[i] = final_parameters[j]
    #        j += 1

    #labels.append('vrot')
    #labels.append('vmac')
    #labels.append('doppler_shift')
    j = 0
    for label, value in final_parameters.items():
        std_error = final_parameters_std[label]
        if label != 'teff':
            print(f"{label:<15}: {value:>10.3f} +/- {std_error:>10.3f}")
        else:
            print(f"{label:<15}: {value*1000:>10.3f} +/- {std_error*1000:>10.3f}")
    if resolution_val is not None:
        print(f"{'Resolution':<15}: {int(resolution_val):>10}")


    doppler_shift = final_parameters['doppler_shift']
    #vmac = final_parameters['vmac']
    vmac = 0
    vrot = final_parameters['vsini']

    final_params = []

    for label in labels:
        final_params.append(final_parameters[label])

    real_labels = final_params
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
    #plt.plot(wavelength_test * (1 + (doppler_shift / 299792.)) * (1 + (doppler_shift / 299792.)), flux_test, label="Payne test", color='b')
    plt.ylim(0.0, 1.05)
    plt.xlim(wavelength_payne_plot[0], wavelength_payne_plot[-1])
    plt.show()