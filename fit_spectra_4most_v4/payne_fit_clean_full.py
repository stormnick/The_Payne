from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy
import spectral_model
from convolve import *
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 03.03.25

def process_spectra(wavelength_payne, wavelength_obs, flux_obs, h_line_cores, h_line_core_mask_dlam=0.5, extra_payne_cut=10):
    """
    Loads the observed spectrum, cuts out unnecessary parts, and processes it to be used with the Payne model.
    :param wavelength_payne: Wavelength array corresponding to the Payne model. Rest frame.
    :param wavelength_obs: Wavelengths of the observed spectrum. Not in the rest frame.
    :param flux_obs: Fluxes of the observed spectrum.
    :param h_line_cores: List of hydrogen line core wavelengths to mask out.
    :param h_line_core_mask_dlam: The width of the mask around the hydrogen line cores in Angstroms.
    :param extra_payne_cut: Extra cut around the Payne model wavelength range in Angstroms.
    :return: Returns the processed wavelength and flux arrays.
    """
    # remove any negative fluxes and fluxes > 1.2
    mask = (flux_obs > 0.0) & (flux_obs < 1.2)
    wavelength_obs = wavelength_obs[mask]
    flux_obs = flux_obs[mask]

    # mask out the hydrogen line cores; to be fair not fully correct because h_line_cores are in rest frame, but we
    # assume that the observed spectrum is close enough to the rest frame
    mask = np.all(np.abs(wavelength_obs[:, None] - h_line_cores) > h_line_core_mask_dlam, axis=1)
    wavelength_obs = wavelength_obs[mask]
    flux_obs = flux_obs[mask]

    # cut the observed spectrum to the Payne model wavelength range; no need to carry around the whole spectrum
    l_cut = (wavelength_obs > wavelength_payne[0] - extra_payne_cut) & (wavelength_obs < wavelength_payne[-1] + extra_payne_cut)
    wavelength_obs = wavelength_obs[l_cut]
    flux_obs = flux_obs[l_cut]
    return wavelength_obs, flux_obs

def calculate_vturb(teff: float, logg: float, met: float) -> float:
    """
    Calculates micro turbulence based on the input parameters
    :param teff: Temperature in kelvin
    :param logg: log(g) in dex units
    :param met: metallicity [Fe/H] scaled by solar
    :return: micro turbulence in km/s
    """
    t0 = 5500.
    g0 = 4.
    tlim = 5000.
    glim = 3.5

    delta_logg = logg - g0

    if logg >= glim:
        # dwarfs
        if teff >= tlim:
            # hot dwarfs
            delta_t = teff - t0
        else:
            # cool dwarfs
            delta_t = tlim - t0

        v_mturb = (1.05 + 2.51e-4 * delta_t + 1.5e-7 * delta_t**2 - 0.14 * delta_logg - 0.005 * delta_logg**2 +
                   0.05 * met + 0.01 * met**2)

    elif logg < glim:
        # giants
        delta_t = teff - t0

        v_mturb = (1.25 + 4.01e-4 * delta_t + 3.1e-7 * delta_t**2 - 0.14 * delta_logg - 0.005 * delta_logg**2 +
                   0.05 * met + 0.01 * met**2)

    return v_mturb

def apply_on_segments(
        wavelength,            # 1-D array-like
        spectrum,              # 1-D array-like, same length
        func,                  # callable(wl_seg, sp_seg, *a, **kw) → (wl_out, sp_out)
        *func_args,            # forwarded positional args (e.g. vsini)
        spacing_tolerance=2.0, # gap = diff > tol × median_spacing
        assume_sorted=True,    # set False if the array is unsorted
        **func_kwargs):        # forwarded keyword args
    """
    Call `func` independently on each uniformly-spaced wavelength segment.
    Basically, it applies convolution on each spectra segment that is separated by a "large" gap.
    It is faster than applying `func` on the whole array at once, especially for large arrays.

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
    """
    Loads the Payne model coefficients from a .npz file.
    :param path_model: Path to the .npz file containing the Payne model coefficients.
    :return: Returns a tuple containing the coefficients and the wavelength array.
    """
    tmp = np.load(path_model)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    w_array_3 = tmp["w_array_3"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    b_array_3 = tmp["b_array_3"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    # wavelength is the wavelength array corresponding to the Payne model in AA
    wavelength = tmp["wavelength"]
    # labels are the label names corresponding to the Payne model, e.g. "teff", "logg", "feh", etc.
    labels = list(tmp["label_names"])
    tmp.close()
    # w_array are the weights, b_array are the biases
    # x_min and x_max are the minimum and maximum values for scaling the labels
    payne_coeffs = (w_array_0, w_array_1, w_array_2, w_array_3,
                    b_array_0, b_array_1, b_array_2, b_array_3,
                    x_min, x_max)
    return payne_coeffs, wavelength, labels


def make_model_spectrum_for_curve_fit(payne_coeffs, wavelength_payne, input_values, resolution_val=None,
            pixel_limits=None, flux_obs=None):
    """
    Creates a model spectrum function for curve fitting.
    :param payne_coeffs: Coefficients from the Payne model. Typically first few arrays are weights and biases,
    while the last two are x_min and x_max for scaling.
    :param wavelength_payne: Wavelength array corresponding to the Payne model. Rest frame.
    :param input_values: Since we cannot pass arguments, we pass values that we fix (i.e. do not fit) in the input_values.
    Example: [10, None, 5] would mean that the first parameter is fixed at 10, the second is free to fit, and the third is fixed at 5.
    :param resolution_val: Resolution value for convolution of Payne spectrum, if applicable.
    :param pixel_limits: A boolean mask for the pixel limits, if applicable. It masks the payne array to only
    include pixels that are within the limits.
    It can take three different types:
    1. Full of lists/tuples with wavelength limits, e.g. [(5000, 5005), (6000, 6005)] would mean that only pixels
    between 5000 and 5005 and between 6000 and 6005 are used.
    2. length is 2, with limits where to use: E.g. [5000, 5005] would mean that only pixels between 5000 and 5005 are used.
    3. Same length as the wavelength_payne, mask of the pixels to mask: E.g. [True, False, True] would mean that the
    first and third pixels are included, while the second is excluded.
    :param flux_obs: Observations, but only for plotting purposes, not for fitting.
    :return: Callable function that takes wavelength_obs and parameters to fit, and returns the model spectrum.
    """
    def model_spectrum_for_curve_fit(wavelength_obs, *params_to_fit):
        """
        Model spectrum function for curve fitting.
        :param wavelength_obs: Wavelengths of the observed spectrum. Not in the rest frame, but the closer it is, the better
        :param params_to_fit: A list of parameters to fit. The length of this list should be equal to the number of
        parameters that are not fixed in the input_values.
        Example: if input_values = [10, None, 5], then params_to_fit should have length 1, and it will be the second parameter
        and it will be fitted.
        :return: Interpolated model spectrum at the observed wavelengths (i.e not in the rest frame).
        """
        # spectra_params is what we will use to create the model spectrum. Take input values from `input_values` that we
        # do not fit and fill in the None values with the values we fit from `params_to_fit`.
        spectra_params = np.array(input_values).copy().astype(float)
        j = 0
        for i, input_value in enumerate(input_values):
            if input_value is None:
                spectra_params[i] = params_to_fit[j]
                j += 1

        # here we assume that the last three parameters are vsini, vmac, and doppler_shift
        vsini = spectra_params[-3]
        vmac = spectra_params[-2]
        doppler_shift = spectra_params[-1]

        real_labels = spectra_params[:-3]

        # if vmic is 99, then scale with teff/logg/feh
        # if we pass vmic = 99 (i.e. not None), it is not fitted. But sometimes we want to calculate it based on the
        # emprirical formula for the speed. Citation: Bergemann & Hoppe (LRCA, in prep.)
        if real_labels[3] >= 99:
            real_labels[3] = calculate_vturb(real_labels[0] * 1000, real_labels[1], real_labels[2])

        # scale the labels to the Payne coefficients and get the spectrum
        scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
        spec_payne = spectral_model.get_spectrum_from_neural_net(
            scaled_labels=scaled_labels,
            NN_coeffs=payne_coeffs,
            pixel_limits=pixel_limits
        )

        wavelength_payne_ = wavelength_payne

        # debug: preview the spectrum before convolutions
        #plt.figure(figsize=(14, 7))
        #plt.title(params_to_fit[0])
        #plt.scatter(wavelength_obs, flux_obs, s=3, color='k')
        #plt.plot(wavelength_payne_, spec_payne, color='r')
        #plt.show()

        # apply convolutions if needed
        if vmac > 0:
            wavelength_payne_, spec_payne = apply_on_segments(
                wavelength_payne_,
                spec_payne,
                conv_macroturbulence,
                vmac
            )
        if vsini > 0:
            wavelength_payne_, spec_payne = apply_on_segments(
                wavelength_payne_,
                spec_payne,
                conv_rotation,
                vsini
            )
        if resolution_val is not None:
            wavelength_payne_, spec_payne = conv_res(wavelength_payne_, spec_payne, resolution_val)

        # apply doppler shift
        if doppler_shift != 0:
            wavelength_payne_ = wavelength_payne_ * (1 + (doppler_shift / 299792.))

        # debug: preview the spectrum after convolutions
        #plt.figure(figsize=(14, 7))
        #plt.title(params_to_fit[0])
        #plt.scatter(wavelength_obs, flux_obs, s=3, color='k')
        #plt.plot(wavelength_payne_, spec_payne, color='r')
        #plt.show()

        # interpolate the spectrum to the observed wavelengths
        f_interp = interp1d(
            wavelength_payne_,
            spec_payne,
            kind='linear',
            bounds_error=False,
            fill_value=1
        )
        interpolated_spectrum = f_interp(wavelength_obs)

        # debug: preview the interpolated spectrum
        # calculate chi-squared
        if flux_obs is not None and False:
            chi_squared = np.sum((interpolated_spectrum - flux_obs) ** 2)
            print(params_to_fit[0], chi_squared)
            plt.figure(figsize=(14, 7))
            plt.title(params_to_fit[0])
            plt.scatter(wavelength_obs, flux_obs, s=3, color='k')
            plt.scatter(wavelength_obs, interpolated_spectrum, s=2, color='r')
            plt.show()

        return interpolated_spectrum

    return model_spectrum_for_curve_fit

def scale_back(x, x_min, x_max, label_name=None):
    """
    Scales back the input values from the Payne model to the original scale.
    :param x: Values to scale back, typically coefficients from the Payne model.
    :param x_min: Minimum values for scaling, corresponding to the Payne model.
    :param x_max: Maximum values for scaling, corresponding to the Payne model.
    :param label_name: Optional label name to adjust the scaling for specific labels, e.g. "teff".
    :return: Returns the scaled back values as a list.
    """
    x = np.array(x)
    x_min = np.array(x_min)
    x_max = np.array(x_max)
    return_value = (x + 0.5) * (x_max - x_min) + x_min
    # if teff, the teff is typically in kK, so we scale it back to K
    if label_name == "teff":
        return_value = return_value * 1000
    return list(return_value)


def get_bounds_and_p0(p0, input_values, def_bounds, labels):
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

    return p0, columns_to_pop, labels_to_fit, def_bounds


def cut_to_just_lines(wavelength_obs, flux_obs, wavelength_payne, lines_to_use, stellar_rv, obs_cut_aa=0.5, payne_cut_aa=0.75):
    # cut so that we only take the lines we want
    masks = []
    masks_payne = []

    if type(obs_cut_aa) is not list:
        obs_cut_aa = [obs_cut_aa] * len(lines_to_use)
    if type(payne_cut_aa) is not list:
        payne_cut_aa = [payne_cut_aa] * len(lines_to_use)

    if len(obs_cut_aa) == 1:
        obs_cut_aa = obs_cut_aa * len(lines_to_use)
    if len(payne_cut_aa) == 1:
        payne_cut_aa = payne_cut_aa * len(lines_to_use)

    wavelength_obs_rv_corrected = wavelength_obs / (1 + (stellar_rv / 299792.))

    for line, obs_cut_aa_one, payne_cut_aa_one in zip(lines_to_use, obs_cut_aa, payne_cut_aa):
        mask_one = (wavelength_obs_rv_corrected > line - obs_cut_aa_one) & (wavelength_obs_rv_corrected < line + obs_cut_aa_one)
        masks.append(mask_one)

        mask_payne = (wavelength_payne > line - payne_cut_aa_one) & (wavelength_payne < line + payne_cut_aa_one)
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
    #p0[4:] = (len(p0) - 4) * [0.5]
    # add extra 3 0s
    p0 += [3, 3, 0]

    input_values = [None] * len(p0)
    def_bounds = (x_min + [0, 0, -150 + stellar_rv], x_max + [100, 100, 150 + stellar_rv])
    return p0, input_values, def_bounds


def fit_teff(labels, payne_coeffs, x_min, x_max, stellar_rv, teff_lines_to_use, wavelength_obs, flux_obs, wavelength_payne, resolution_val, silent=False):
    p0, input_values, def_bounds = get_default_p0_guess(labels, payne_coeffs, x_min, x_max, stellar_rv)
    input_values = [None, None, None, 99] + [0] * (len(labels) - 4) + [None, 0, None]
    p0, columns_to_pop, labels_to_fit, def_bounds = get_bounds_and_p0(p0, input_values, def_bounds, labels)
    wavelength_obs_cut_to_lines, flux_obs_cut_to_lines, wavelength_payne_cut, combined_mask_payne = cut_to_just_lines(
        wavelength_obs, flux_obs, wavelength_payne, teff_lines_to_use, stellar_rv, obs_cut_aa=0.75, payne_cut_aa=1)
    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne_cut,
        input_values,
        resolution_val=resolution_val,
        pixel_limits=combined_mask_payne
    )
    if not silent:
        print("Fitting...")
        time_start = time.perf_counter()
    popt, pcov = curve_fit(
        model_func,
        wavelength_obs_cut_to_lines,
        flux_obs_cut_to_lines,
        p0=p0,
        bounds=def_bounds,
    )
    if not silent:
        print(f"Done fitting in {time.perf_counter() - time_start:.2f} seconds")
        print(f"Fitted teff: {popt[0] * 1000:.3f} +/- {np.sqrt(np.diag(pcov))[0]:.3f}")
        print(popt)
    return float(popt[0]), float(np.sqrt(np.diag(pcov))[0])

def fit_logg(final_parameters, labels, payne_coeffs, x_min, x_max, stellar_rv, wavelength_obs, flux_obs, wavelength_payne, resolution_val, silent=False):
    # load mg_fe and ca_fe lines
    mg_fe_lines = pd.read_csv("../linemasks/mg_triplet.csv")
    mg_fe_lines = mg_fe_lines['ll']
    ca_fe_lines = pd.read_csv("../linemasks/ca_triplet.csv")
    ca_fe_lines = ca_fe_lines['ll']
    fe_lines = pd.read_csv("../fe_lines_hr_good.csv")
    fe_lines = list(fe_lines["ll"])
    # combine both
    logg_lines = list(mg_fe_lines) + list(fe_lines) + list(ca_fe_lines)
    p0, input_values, def_bounds = get_default_p0_guess(labels, payne_coeffs, x_min, x_max, stellar_rv)
    input_values = [final_parameters["teff"], None, None, 99] + [0] * (len(labels) - 4) + [None, 0, None]
    # find location of mg_fe and ca_fe in the labels
    mg_index = labels.index("Mg_Fe")
    ca_index = labels.index("Ca_Fe")
    # set mg and ca to 0
    input_values[mg_index] = None
    input_values[ca_index] = None
    p0, columns_to_pop, labels_to_fit, def_bounds = get_bounds_and_p0(p0, input_values, def_bounds, labels)
    wavelength_obs_cut_to_lines, flux_obs_cut_to_lines, wavelength_payne_cut, combined_mask_payne = cut_to_just_lines(
        wavelength_obs, flux_obs, wavelength_payne, logg_lines, stellar_rv, obs_cut_aa=1, payne_cut_aa=2)
    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne_cut,
        input_values,
        resolution_val=resolution_val,
        pixel_limits=combined_mask_payne
    )
    if not silent:
        print("Fitting...")
        time_start = time.perf_counter()
    popt, pcov = curve_fit(
        model_func,
        wavelength_obs_cut_to_lines,
        flux_obs_cut_to_lines,
        p0=p0,
        bounds=def_bounds,
    )
    if not silent:
        print(f"Done fitting in {time.perf_counter() - time_start:.2f} seconds")
        print(f"Fitted logg: {popt[0]:.3f} +/- {np.sqrt(np.diag(pcov))[0]:.3f}")
        print(f"Fitted doppler shift: {popt[-1]:.3f} +/- {np.sqrt(np.diag(pcov))[-1]:.3f}")
    return float(popt[0]), float(np.sqrt(np.diag(pcov))[0]), float(popt[-1]), float(np.sqrt(np.diag(pcov))[-1])

def fit_teff_logg(labels, payne_coeffs, x_min, x_max, stellar_rv, wavelength_obs, flux_obs, wavelength_payne, resolution_val, do_hydrogen_lines=False, silent=False, p0_input=None):
    # load mg_fe and ca_fe lines
    h_line_cores = pd.read_csv("../linemasks/h_cores.csv")
    h_line_cores = h_line_cores['ll']
    h_line_cut = [15] * len(h_line_cores)
    h_line_payne_cut = [20] * len(h_line_cores)
    mg_fe_lines = pd.read_csv("../linemasks/mg_triplet.csv")
    mg_fe_lines = mg_fe_lines['ll']
    mg_line_cut = [1] * len(mg_fe_lines)
    mg_line_payne_cut = [1.25] * len(mg_fe_lines)
    ca_fe_lines = pd.read_csv("../linemasks/ca_triplet.csv")
    ca_fe_lines = ca_fe_lines['ll']
    ca_line_cut = [1] * len(ca_fe_lines)
    ca_line_payne_cut = [1.25] * len(ca_fe_lines)
    fe_lines = pd.read_csv("../linemasks/fe_lines_hr_good.csv")
    fe_lines = list(fe_lines["ll"])
    fe_line_cut = [1] * len(fe_lines)
    fe_line_payne_cut = [1.25] * len(fe_lines)
    # combine both
    logg_lines = list(mg_fe_lines) + list(ca_fe_lines) + list(fe_lines)
    p0, input_values, def_bounds = get_default_p0_guess(labels, payne_coeffs, x_min, x_max, stellar_rv)
    input_values = [None, None, None, None] + [0] * (len(labels) - 4) + [None, 0, None]
    # find location of mg_fe and ca_fe in the labels
    mg_index = labels.index("Mg_Fe")
    ca_index = labels.index("Ca_Fe")
    o_index = labels.index("O_Fe")
    c_index = labels.index("C_Fe")
    # set mg and ca to 0
    input_values[mg_index] = None
    input_values[ca_index] = None
    #input_values[o_index] = None
    #input_values[c_index] = None

    lines_to_use = mg_line_cut + ca_line_cut + fe_line_cut
    lines_to_cut = mg_line_payne_cut + ca_line_payne_cut + fe_line_payne_cut

    if do_hydrogen_lines:
        lines_to_use += h_line_cut
        lines_to_cut += h_line_payne_cut
        logg_lines += list(h_line_cores)

    p0, columns_to_pop, labels_to_fit, def_bounds = get_bounds_and_p0(p0, input_values, def_bounds, labels)

    if p0_input is not None:
        p0 = p0_input

    wavelength_obs_cut_to_lines, flux_obs_cut_to_lines, wavelength_payne_cut, combined_mask_payne = cut_to_just_lines(
        wavelength_obs, flux_obs, wavelength_payne, logg_lines, stellar_rv, obs_cut_aa=lines_to_use,
        payne_cut_aa=lines_to_cut)
    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne_cut,
        input_values,
        resolution_val=resolution_val,
        pixel_limits=combined_mask_payne
    )
    if not silent:
        print("Fitting...")
        time_start = time.perf_counter()
    popt, pcov = curve_fit(
        model_func,
        wavelength_obs_cut_to_lines,
        flux_obs_cut_to_lines,
        p0=p0,
        bounds=def_bounds,
        max_nfev=10e5
    )
    print(popt)
    if not silent:
        print(f"Done fitting in {time.perf_counter() - time_start:.2f} seconds")
        print(f"Fitted teff: {popt[0]:.3f} +/- {np.sqrt(np.diag(pcov))[0]:.3f}")
        print(f"Fitted logg: {popt[1]:.3f} +/- {np.sqrt(np.diag(pcov))[1]:.3f}")
        print(f"Fitted doppler shift: {popt[-1]:.3f} +/- {np.sqrt(np.diag(pcov))[-1]:.3f}")
    return float(popt[0]), float(np.sqrt(np.diag(pcov))[0]), float(popt[1]), float(np.sqrt(np.diag(pcov))[1]), float(popt[-1]), float(np.sqrt(np.diag(pcov))[-1]), popt


def fit_feh(final_parameters, labels, payne_coeffs, x_min, x_max, stellar_rv, wavelength_obs, flux_obs, wavelength_payne, resolution_val, silent=False, fit_vsini=False, fit_vmac=False):
    fe_lines = pd.read_csv("../linemasks/fe_lines_hr_good.csv")
    fe_lines = list(fe_lines["ll"])
    p0, input_values, def_bounds = get_default_p0_guess(labels, payne_coeffs, x_min, x_max, stellar_rv)

    if fit_vsini:
        vsini_value = None
    else:
        vsini_value = 0

    if fit_vmac:
        vmac_value = None
    else:
        vmac_value = 0

    input_values = [final_parameters["teff"], final_parameters["logg"], None, None] + [0] * (
                len(labels) - 4) + [vsini_value, vmac_value, final_parameters["doppler_shift"]]

    o_index = labels.index("O_Fe")
    c_index = labels.index("C_Fe")
    input_values[o_index] = None
    input_values[c_index] = None

    p0, columns_to_pop, labels_to_fit, def_bounds = get_bounds_and_p0(p0, input_values, def_bounds, labels)

    wavelength_obs_cut_to_lines, flux_obs_cut_to_lines, wavelength_payne_cut, combined_mask_payne = cut_to_just_lines(
        wavelength_obs, flux_obs, wavelength_payne, fe_lines, stellar_rv, obs_cut_aa=0.5, payne_cut_aa=0.75)

    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne_cut,
        input_values,
        resolution_val=resolution_val,
        pixel_limits=combined_mask_payne
    )

    if not silent:
        print("Fitting...")
        time_start = time.perf_counter()

    popt, pcov = curve_fit(
        model_func,
        wavelength_obs_cut_to_lines,
        flux_obs_cut_to_lines,
        p0=p0,
        bounds=def_bounds,
        max_nfev=10e5
    )

    if fit_vsini and fit_vmac:
        vsini_value = float(popt[-2])
        vsini_error = float(np.sqrt(np.diag(pcov))[-2])
        vmac_value = float(popt[-1])
        vmac_error = float(np.sqrt(np.diag(pcov))[-1])
    elif fit_vsini and not fit_vmac:
        vsini_value = float(popt[-1])
        vsini_error = float(np.sqrt(np.diag(pcov))[-1])
        vmac_value = 0
        vmac_error = -1
    elif fit_vmac and not fit_vsini:
        vmac_value = float(popt[-1])
        vmac_error = float(np.sqrt(np.diag(pcov))[-1])
        vsini_value = 0
        vsini_error = -1
    else:
        vmac_value, vmac_error = 0, -1
        vsini_value, vsini_error = 0, -1

    if not silent:
        print(f"Done fitting in {time.perf_counter() - time_start:.2f} seconds")
        print(f"Fitted feh: {popt[0]:.3f} +/- {np.sqrt(np.diag(pcov))[0]:.3f}")
        print(f"Fitted vmic: {popt[1]:.3f} +/- {np.sqrt(np.diag(pcov))[1]:.3f}")
        print(f"Fitted vsini: {vsini_value:.3f} +/- {vsini_error:.3f}")
        print(f"Fitted vmac: {vmac_value:.3f} +/- {vmac_error:.3f}")

    return float(popt[0]), float(np.sqrt(np.diag(pcov))[0]), float(popt[1]), float(np.sqrt(np.diag(pcov))[1]), vsini_value, vsini_error, vmac_value, vmac_error

def scale_dlam(dlam, broadening):
    # scales dlam where to fit a line with broadening, a rough calculation based on my own
    # done for R = 20 000
    # at 0 broadening scaling ~ 1
    # at 10 scaling ~ 1.2
    # at 50 scaling ~ 2.8
    # at 100 scaling ~ 5

    # quadratic scaling
    coefficients = [5.7e-05, 0.014481, 0.472108]
    scaling = (coefficients[0] * broadening ** 2 + coefficients[1] * broadening + coefficients[2]) * 2

    if type(dlam) is list:
        dlam = np.asarray(dlam)

    return dlam * scaling


def fit_one_xfe_element(final_parameters, element_to_fit, labels, payne_coeffs, x_min, x_max, stellar_rv, wavelength_obs, flux_obs, wavelength_payne, resolution_val, silent=False):
    if element_to_fit == "A_Li":
        path = f"../linemasks/li.csv"
    else:
        path = f"../linemasks/{element_to_fit.split('_')[0].lower()}.csv"
    if os.path.exists(path):
        elements_lines_data = pd.read_csv(path)
        element_lines = list(elements_lines_data['ll'])
        if "dlam" in elements_lines_data.columns:
            dlam = elements_lines_data['dlam'].tolist()
        else:
            dlam = [0.5]
    else:
        element_lines = [5000]
        dlam = [2000]

    dlam = scale_dlam(dlam, final_parameters["vsini"])

    p0, input_values, def_bounds = get_default_p0_guess(labels, payne_coeffs, x_min, x_max, stellar_rv)
    input_values = [(final_parameters["teff"]), (final_parameters["logg"]), (final_parameters["feh"]),
                    (final_parameters["vmic"])] + [0] * (len(labels) - 4) + [(final_parameters["vsini"]), 0,
                                                                                  (final_parameters[
                                                                                            "doppler_shift"])]

    for fitted_label in final_parameters.keys():
        if fitted_label.endswith("_Fe") and fitted_label in labels:
            index = labels.index(fitted_label)
            input_values[index] = final_parameters[fitted_label]

    index_element = labels.index(element_to_fit)
    input_values[index_element] = None

    p0, columns_to_pop, labels_to_fit, def_bounds = get_bounds_and_p0(p0, input_values, def_bounds, labels)

    wavelength_obs_cut_to_lines, flux_obs_cut_to_lines, wavelength_payne_cut, combined_mask_payne = cut_to_just_lines(
        wavelength_obs, flux_obs, wavelength_payne, element_lines, stellar_rv, obs_cut_aa=list(dlam), payne_cut_aa=list(np.asarray(dlam) * 1.5))

    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne_cut,
        input_values,
        resolution_val=resolution_val,
        pixel_limits=combined_mask_payne,
        flux_obs = flux_obs_cut_to_lines
    )

    if not silent:
        print("Fitting...")
        time_start = time.perf_counter()

    try:
        popt, pcov = curve_fit(
            model_func,
            wavelength_obs_cut_to_lines,
            flux_obs_cut_to_lines,
            p0=p0,
            bounds=def_bounds,
        )
    except ValueError:
        print(f"Fitting failed for element {element_to_fit}")
        return -99, -1

    if not silent:
        print(f"Done fitting in {time.perf_counter() - time_start:.2f} seconds")
        print(f"Fitted {element_to_fit}: {popt[0]:.3f} +/- {np.sqrt(np.diag(pcov))[0]:.3f}")

    #input_values[index_element] = float(popt[0])
    return float(popt[0]), float(np.sqrt(np.diag(pcov))[0])

def plot_fitted_payne(wavelength_payne, final_parameters, payne_coeffs, wavelength_obs, flux_obs, labels, resolution_val=None, real_labels2=None, real_labels2_xfe=None, real_labels2_vsini=None, plot_show=True):
    doppler_shift = final_parameters['doppler_shift']
    vmac = final_parameters['vmac']
    vsini = final_parameters['vsini']

    final_params = []

    for label in labels:
        final_params.append(final_parameters[label])

    real_labels = final_params

    scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
    payne_fitted_spectra = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels,
                                                                       NN_coeffs=payne_coeffs)

    wavelength_payne_plot = wavelength_payne
    if vmac > 1e-3:
        wavelength_payne_plot, payne_fitted_spectra = conv_macroturbulence(wavelength_payne_plot,
                                                                           payne_fitted_spectra, vmac)
    if vsini > 1e-3:
        wavelength_payne_plot, payne_fitted_spectra = conv_rotation(wavelength_payne_plot, payne_fitted_spectra,
                                                                    vsini)
    if resolution_val is not None:
        wavelength_payne_plot, payne_fitted_spectra = conv_res(wavelength_payne_plot, payne_fitted_spectra,
                                                               resolution_val)


    if real_labels2 is not None:
        real_labels[0:4] = real_labels2
        if real_labels2_xfe is not None:
            for element in real_labels2_xfe.keys():
                idx_element = labels.index(element)
                real_labels[idx_element] = real_labels2_xfe[element]
        scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
        payne_fitted_spectra2 = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels,
                                                                            NN_coeffs=payne_coeffs)

        wavelength_payne_plot2 = wavelength_payne
        if vmac > 1e-3:
            wavelength_payne_plot2, payne_fitted_spectra2 = conv_macroturbulence(wavelength_payne_plot2,
                                                                                 payne_fitted_spectra2, vmac)
        if vsini > 1e-3 and real_labels2_vsini is None:
            wavelength_payne_plot2, payne_fitted_spectra2 = conv_rotation(wavelength_payne_plot2, payne_fitted_spectra2,
                                                                          vsini)
        elif real_labels2_vsini is not None:
            wavelength_payne_plot2, payne_fitted_spectra2 = conv_rotation(wavelength_payne_plot2, payne_fitted_spectra2, real_labels2_vsini)
        if resolution_val is not None:
            wavelength_payne_plot2, payne_fitted_spectra2 = conv_res(wavelength_payne_plot2, payne_fitted_spectra2,
                                                                     resolution_val)


    # cut wavelength_obs to the payne windows
    windows = [[3926, 4355], [5160, 5730], [6100, 6790]]
    wavelength_obs_cut = []
    flux_obs_cut = []
    for window in windows:
        mask = (wavelength_obs > window[0]) & (wavelength_obs < window[1])
        wavelength_obs_cut.extend(wavelength_obs[mask])
        flux_obs_cut.extend(flux_obs[mask])

    #np.savetxt("fitted_spectrum_.txt", np.vstack((wavelength_payne_plot * (1 + (doppler_shift / 299792.)), payne_fitted_spectra)).T, fmt='%.6f', header='wavelength_payne flux_payne')

    plt.figure(figsize=(18, 6))
    plt.scatter(wavelength_obs_cut, flux_obs_cut, label="Observed", s=3, color='k')
    plt.plot(wavelength_payne_plot * (1 + (doppler_shift / 299792.)), payne_fitted_spectra, label="Payne",
             color='r')
    if real_labels2 is not None:
        plt.plot(wavelength_payne_plot2 * (1 + (doppler_shift / 299792.)), payne_fitted_spectra2, label="Payne input",
                 color='b')
    plt.legend()
    plt.ylim(0.0, 1.05)
    plt.xlim(wavelength_payne_plot[0], wavelength_payne_plot[-1])
    mg_fe_lines = pd.read_csv("../linemasks/mg_triplet.csv")
    mg_fe_lines = mg_fe_lines['ll']
    ca_fe_lines = pd.read_csv("../linemasks/ca_triplet.csv")
    ca_fe_lines = ca_fe_lines['ll']
    # combine both
    logg_lines = list(mg_fe_lines) + list(ca_fe_lines)
    for logg_line in logg_lines:
        plt.axvline(logg_line, color='grey')
    if plot_show:
        plt.show()


if __name__ == '__main__':
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_alt_smallerldelta_ts_nlte_lesselements_hr10_2025-02-27-08-43-08.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr3_2025-03-10-10-19-24.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr10_2025-03-12-07-46-13.npz"
    #path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr13_2025-03-12-08-29-38.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr15n_2025-03-12-08-32-42.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr15n_smalldelta_2025-03-24-14-04-40.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr15n_smalldelta_2025-03-24-14-46-44.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_2025-03-27-08-06-34.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_2025-06-06-14-18-21.npz"
    #path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr10_2025-03-12-07-46-13.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_batch01_medium_test2training_reducedlogg_altarch_2025-06-16-06-28-26.npz"
    """teff           :   5692.693 +/-      3.459
logg           :      4.469 +/-      0.013
feh            :     -0.261 +/-      0.004
vmic           :      1.385 +/-      0.014
A_Li           :      0.000 +/-      0.218
C_Fe           :      0.347 +/-      0.008
Ca_Fe          :      0.037 +/-      0.009
Ba_Fe          :     -0.058 +/-      0.021
vsini          :      0.000 +/-     -1.000
vmac           :      2.216 +/-      0.024
doppler_shift  :      0.327 +/-      0.008"""
    payne_coeffs, wavelength_payne, labels = load_payne(path_model)
    x_min = list(payne_coeffs[-2])
    x_max = list(payne_coeffs[-1])

    label_names = labels.copy()
    label_names.append('vsini')
    label_names.append('vmac')
    label_names.append('doppler_shift')

    resolution_val = None

    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/KPNO_FTS_flux_2960_13000_Kurucz1984.txt", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/melchiors.txt", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/melchiors.txt", dtype=float, unpack=True)
    wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PycharmProjects/payne/observed_spectra_to_test/Sun_melchiors_spectrum.txt", dtype=float, unpack=True)
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

    #folder = "/Users/storm/Downloads/Cont/"
    #folder_spectra = "/Users/storm/Downloads/Science/"
    #file1 = "Sun_melchiors_spectrum.npy"
    #continuum = np.load(f"{folder}{file1}")
    #spectra = np.load(f"{folder_spectra}{file1}")
    #wavelength_obs = continuum[0]
    #flux_obs = spectra[1] / continuum[1]

    #wavelength_obs, flux_obs = conv_res(wavelength_obs, flux_obs, 20000)

    h_line_cores = pd.read_csv("../linemasks/h_cores.csv")
    h_line_cores = list(h_line_cores['ll'])

    wavelength_obs, flux_obs = process_spectra(wavelength_payne, wavelength_obs, flux_obs, h_line_cores, h_line_core_mask_dlam=0.5)

    final_parameters = {}
    final_parameters_std = {}

    teff, teff_std, logg, logg_std, doppler_shift, doppler_shift_std, popt = fit_teff_logg(labels, payne_coeffs, x_min,
                                                                                           x_max, stellar_rv,
                                                                                           wavelength_obs, flux_obs,
                                                                                           wavelength_payne,
                                                                                           resolution_val, silent=False)

    final_parameters["teff"] = teff
    final_parameters_std["teff"] = teff_std
    final_parameters["logg"] = logg
    final_parameters["doppler_shift"] = doppler_shift
    final_parameters_std["logg"] = logg_std
    final_parameters_std["doppler_shift"] = doppler_shift_std

    # 3. FEH, VMIC, VMAC
    feh, feh_std, vmic, vmic_std, vsini, vsini_std, vmac, vmac_std = fit_feh(final_parameters, labels, payne_coeffs,
                                                                             x_min, x_max, stellar_rv, wavelength_obs,
                                                                             flux_obs, wavelength_payne, resolution_val,
                                                                             silent=False, fit_vsini=True, fit_vmac=False)

    final_parameters["feh"] = feh
    final_parameters["vmic"] = vmic
    final_parameters["vsini"] = vsini
    final_parameters["vmac"] = vmac
    final_parameters_std["feh"] = feh_std
    final_parameters_std["vmic"] = vmic_std
    final_parameters_std["vsini"] = vsini_std
    final_parameters_std["vmac"] = vmac_std

    # 4. REMAINING ELEMENTS ONE-BY-ONE
    # find how many _Fe labels are there
    elements_to_fit = []
    for i, label in enumerate(labels):
        if label.endswith("_Fe") or label == "A_Li":
            elements_to_fit.append(label)

    for element_to_fit in elements_to_fit:
        xfe, xfe_std = fit_one_xfe_element(final_parameters, element_to_fit, labels, payne_coeffs, x_min, x_max, stellar_rv, wavelength_obs, flux_obs,
                            wavelength_payne, resolution_val, silent=False)

        final_parameters[element_to_fit] = xfe
        final_parameters_std[element_to_fit] = xfe_std


    # PRINT RESULTS
    for label in label_names:
        value = final_parameters[label]
        std_error = final_parameters_std[label]
        if label != 'teff':
            print(f"{label:<15}: {value:>10.3f} +/- {std_error:>10.3f}")
        else:
            print(f"{label:<15}: {value*1000:>10.3f} +/- {std_error*1000:>10.3f}")
    if resolution_val is not None:
        print(f"{'Resolution':<15}: {int(resolution_val):>10}")


    doppler_shift = final_parameters['doppler_shift']
    vmac = final_parameters['vmac']
    vsini = final_parameters['vsini']

    final_params = []

    for label in labels:
        final_params.append(final_parameters[label])

    real_labels = final_params
    scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
    payne_fitted_spectra = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels,
                                                                  NN_coeffs=payne_coeffs)

    wavelength_payne_plot = wavelength_payne
    if vmac > 1e-3:
        wavelength_payne_plot, payne_fitted_spectra = conv_macroturbulence(wavelength_payne_plot, payne_fitted_spectra, vmac)
    if vsini > 1e-3:
        wavelength_payne_plot, payne_fitted_spectra = conv_rotation(wavelength_payne_plot, payne_fitted_spectra, vsini)
    if resolution_val is not None:
        wavelength_payne_plot, payne_fitted_spectra = conv_res(wavelength_payne_plot, payne_fitted_spectra, resolution_val)

    plt.figure(figsize=(18, 6))
    plt.scatter(wavelength_obs, flux_obs, label="Observed", s=3, color='k')
    plt.plot(wavelength_payne_plot * (1 + (doppler_shift / 299792.)), payne_fitted_spectra, label="Payne", color='r')
    #plt.plot(wavelength_test * (1 + (doppler_shift / 299792.)) * (1 + (doppler_shift / 299792.)), flux_test, label="Payne test", color='b')
    plt.ylim(0.0, 1.05)
    plt.xlim(wavelength_payne_plot[0], wavelength_payne_plot[-1])
    plt.show()