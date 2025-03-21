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
from scipy.optimize import curve_fit, minimize

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


def get_payne_spectra(real_labels, payne_coeffs):
    scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
    spec_payne = spectral_model.get_spectrum_from_neural_net(
        scaled_labels=scaled_labels,
        NN_coeffs=payne_coeffs,
        kovalev_alt=True
    )
    return spec_payne

def make_model_spectrum_for_curve_fit(payne_coeffs, wavelength_payne, input_values, flux_obs, resolution_val=None):
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

        fit_vmac, fit_rotation = False, False

        real_labels = spectra_params[:-3]

        spec_payne = get_payne_spectra(real_labels, payne_coeffs)

        wavelength_payne_ = wavelength_payne

        #if vmac > 0:
        #    wavelength_payne_, spec_payne = conv_macroturbulence(wavelength_payne_, spec_payne, vmac)
        #if vrot > 0:
        #    wavelength_payne_, spec_payne = conv_rotation(wavelength_payne_, spec_payne, vrot)
        #if resolution_val is not None:
        #    wavelength_payne_, spec_payne = conv_res(wavelength_payne_, spec_payne, resolution_val)

        #wavelength_payne_ = wavelength_payne_ * (1 + (doppler_shift / 299792.))

        param_guess, min_bounds = get_rv_macro_rotation_guess(min_macroturb=1,
                                                              max_macroturb=3, fit_vmac=fit_vmac, fit_rotation=fit_rotation)
        # now for the generated abundance it tries to fit best fit macro + doppler shift.
        # Thus, macro should not be dependent on the abundance directly, hopefully
        # Seems to work way better
        # stellar_rv, vmac, rotation, resolution, fit_vmac, fit_rotation, wavelength_obs, flux_norm_obs,
        #                            lmin: float, lmax: float,
        #                            wavelength_fitted: np.ndarray, flux_norm_fitted: np.ndarray
        lmin, lmax = np.min(wavelength_payne_), np.max(wavelength_payne_)
        function_args = (doppler_shift, vmac, vrot, resolution_val, fit_vmac, fit_rotation, wavelength_obs, flux_obs, lmin, lmax, wavelength_payne_, spec_payne)
        minimize_options = {'maxiter': 3 * 50, 'disp': False}
        res = minimize_function(calc_chi_sq_broadening, np.median(param_guess, axis=0),
                                function_args, min_bounds, 'L-BFGS-B', minimize_options)

        rv = res.x[0] + doppler_shift
        if fit_vmac:
            macroturb = res.x[1]
        else:
            macroturb = vmac
        if fit_rotation:
            rotation = res.x[-1]
        else:
            rotation = vrot

        chi_squared = res.fun
        print(f"after fit: vmac={macroturb:.2f}, chisqr={chi_squared:.5f}")

        if resolution_val is not None:
           wavelength_payne_, spec_payne = conv_res(wavelength_payne_, spec_payne, resolution_val)
        if macroturb > 0:
           wavelength_payne_, spec_payne = conv_macroturbulence(wavelength_payne_, spec_payne, macroturb)
        if rotation > 0:
           wavelength_payne_, spec_payne = conv_rotation(wavelength_payne_, spec_payne, rotation)
        wavelength_payne_ = wavelength_payne_ * (1 + (rv / 299792.))

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
        chi_squared = np.sum((interpolated_spectrum - flux_obs) ** 2)
        print(f"teff={real_labels[0]*1000:.2f}, logg={real_labels[1]:.2f}, feh={real_labels[2]:.2f}, chisqr={chi_squared:.5f}")

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

def minimize_function(function_to_minimize, input_param_guess: np.ndarray, function_arguments: tuple, bounds: list[tuple], method: str, options: dict):
    """
    Minimizes a function using specified method and options in the function
    :param function_to_minimize: Function to minimize
    :param input_param_guess: Initial guess for the parameters
    :param function_arguments: Arguments for the function
    :param bounds: Bounds for the parameters
    :param method: Method to use for minimization
    :param options: Options for the minimization
    :return: Result of the minimization
    """
    #res.x: list = [param1 best guess, param2 best guess etc]
    #res.fun: float = function value (chi squared) after the fit

    # using Scipy. Nelder-Mead or L-BFGS- algorithm
    res = minimize(function_to_minimize, input_param_guess, args=function_arguments, bounds=bounds, method=method, options=options)

    return res


def get_convolved_spectra(wave: np.ndarray, flux: np.ndarray, resolution: float, macro: float, rot: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convolves spectra with resolution, macroturbulence or rotation if values are non-zero
    :param wave: wavelength array, in ascending order
    :param flux: flux array normalised
    :param resolution: resolution, zero if not required
    :param macro: Macroturbulence in km/s, zero if not required
    :param rot: Rotation in km/s, 0 if not required
    :return: 2 arrays, first is convolved wavelength, second is convolved flux
    """
    # check that wave and flux are non-empty
    if np.size(wave) == 0 or np.size(flux) == 0:
        return wave, flux
    if resolution != 0.0 and resolution is not None:
        wave_mod_conv, flux_mod_conv = conv_res(wave, flux, resolution)
    else:
        wave_mod_conv = wave
        flux_mod_conv = flux
    if macro != 0.0:
        wave_mod_macro, flux_mod_macro = conv_macroturbulence(wave_mod_conv, flux_mod_conv, macro)
    else:
        wave_mod_macro = wave_mod_conv
        flux_mod_macro = flux_mod_conv
    if rot != 0.0:
        wave_mod, flux_mod = conv_rotation(wave_mod_macro, flux_mod_macro, rot)
    else:
        wave_mod = wave_mod_macro
        flux_mod = flux_mod_macro
    return wave_mod, flux_mod

def calculate_lbl_chi_squared(wave_obs: np.ndarray, flux_obs: np.ndarray, error_obs_variance: np.ndarray,
                              wave_synt_orig: np.ndarray, flux_synt_orig: np.ndarray, resolution: float, lmin: float,
                              lmax: float, vmac: float, rotation: float, convolve_spectra=True) -> float:
    """
    Calculates chi squared by opening a created synthetic spectrum and comparing to the observed spectra. Then
    calculates chi squared. Used for line by line method, by only looking at a specific line.
    :param wave_obs: Observed wavelength
    :param flux_obs: Observed normalised flux
    :param error_obs_variance: Observed error variance
    :param wave_synt_orig: Synthetic wavelength
    :param flux_synt_orig: Synthetic normalised flux
    :param resolution: resolution, zero if not required
    :param lmin: Wavelength, start of line
    :param lmax: Wavelength, end of line
    :param vmac: Macroturbulence in km/s, zero if not required
    :param rotation: Rotation in km/s, 0 if not required
    :return: Calculated chi squared for a given line
    """
    indices_to_use_mod = np.where((wave_synt_orig <= lmax) & (wave_synt_orig >= lmin))
    indices_to_use_obs = np.where((wave_obs <= lmax) & (wave_obs >= lmin))

    wave_synt_orig, flux_synt_orig = wave_synt_orig[indices_to_use_mod], flux_synt_orig[indices_to_use_mod]
    wave_obs, flux_obs = wave_obs[indices_to_use_obs], flux_obs[indices_to_use_obs]
    error_obs_variance = error_obs_variance[indices_to_use_obs]

    if np.size(wave_obs) == 0:
        return 999999

    if convolve_spectra:
        try:
            wave_synt, flux_synt = get_convolved_spectra(wave_synt_orig, flux_synt_orig, resolution, vmac, rotation)
        except ValueError:
            return 999999
    else:
        wave_synt = wave_synt_orig
        flux_synt = flux_synt_orig

    flux_synt_interp = np.interp(wave_obs, wave_synt, flux_synt)

    # replace any zeroes in error_obs_variance with average of the array
    # error_obs_variance is always positive, because we take square of the error
    if np.mean(error_obs_variance) != 0:
        error_obs_variance[error_obs_variance == 0] = np.mean(error_obs_variance)
    else:
        error_obs_variance = np.ones_like(error_obs_variance)

    chi_square = np.sum(np.square(flux_obs - flux_synt_interp) / error_obs_variance) / calculate_dof(wave_obs)
    chi_square = np.sum(np.square(flux_obs - flux_synt_interp))

    return chi_square

def calculate_dof(wave_obs):
    """
    Calculates the degrees of freedom for the given observed wavelength
    :param wave_obs: Observed wavelength
    :return: Degrees of freedom
    """
    # number of points in the observed wavelength
    dof = len(wave_obs) - 5

    return dof

def apply_doppler_correction(wave_ob: np.ndarray, doppler: float) -> np.ndarray:
    return wave_ob / (1 + (doppler / 299792.))

def calc_chi_sq_broadening(param: list, stellar_rv, vmac, rotation, resolution, fit_vmac, fit_rotation, wavelength_obs, flux_norm_obs,
                           lmin: float, lmax: float,
                           wavelength_fitted: np.ndarray, flux_norm_fitted: np.ndarray) -> float:
    """
    Calculates the chi squared for the given broadening parameters and small doppler shift
    :param param: Parameters list with the current evaluation guess
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA] where to calculate chi squared
    :param lmax: End of the line [AA] where to calculate chi squared
    :param wavelength_fitted: Wavelength of synthetic spectra
    :param flux_norm_fitted: Flux of synthetic spectra
    :return: Best fit chi squared
    """
    # param[0] = doppler
    # param[1] = vmac
    # param[-1] = rotation fit

    doppler = stellar_rv + param[0]

    if fit_vmac:
        macroturb = param[1]
    else:
        macroturb = vmac

    if fit_rotation:
        rotation = param[-1]
    else:
        rotation = rotation

    wavelength_obs_rv = apply_doppler_correction(wavelength_obs, doppler)

    error_obs_variance = np.ones_like(flux_norm_obs)

    chi_square = calculate_lbl_chi_squared(wavelength_obs_rv, flux_norm_obs,
                                           error_obs_variance, wavelength_fitted, flux_norm_fitted,
                                           resolution, lmin, lmax, macroturb, rotation)

    return chi_square

def get_simplex_guess(length: int, min_guess: float, max_guess: float, min_bound: float, max_bound: float, guess_random_ratio_to_add=0.1) -> tuple[np.ndarray, tuple]:
    """
    Gets guess if it is fitted for simplex guess
    :param length: number of dimensions (output length+1 array)
    :param min_guess: minimum guess
    :param max_guess: maximum guess
    :param min_bound: minimum bound
    :param max_bound: maximum bound
    :return: Initial guess and minimum bound
    """
    if min_guess < min_bound:
        min_guess = min_bound
    if max_guess > max_bound:
        max_guess = max_bound

    minim_bounds = (min_bound, max_bound)
    # basically adds a bit of randomness to the guess up to this % of the diff of guesses
    guess_difference = np.abs(max_guess - min_guess) * guess_random_ratio_to_add

    initial_guess = np.linspace(min_guess + np.random.random() * guess_difference,
                                max_guess - np.random.random() * guess_difference, length + 1)

    return initial_guess, minim_bounds

def get_rv_macro_rotation_guess(min_rv: float=None, max_rv: float=None, min_macroturb: float=None,
                                max_macroturb: float=None, min_rotation: float=None,
                                max_rotation: float=None, fit_vmac=False, fit_rotation=False) -> tuple[np.ndarray, list[tuple]]:
    """
    Gets rv and macroturbulence guess if it is fitted for simplex guess. np.median(guesses[:, 0]) == 0 is checked
    and if it is 0, then the guess is changed to be halfway between 0 and the max/min, depending whichever is not 0
    Use np.median(guesses, axis=0) to get the median of each parameter for L-BFGS-B minimisation
    :param min_rv: minimum RV for guess (not bounds)
    :param max_rv: maximum RV for guess (not bounds)
    :param min_macroturb: minimum macro for guess (not bounds)
    :param max_macroturb: maximum macro for guess (not bounds)
    :param min_rotation: minimum rotation for guess (not bounds)
    :param max_rotation: maximum rotation for guess (not bounds)
    :return: Initial guess and minimum bound
    """
    # param[0] = rv
    # param[1] = macro IF FITTED
    # param[-1] = rotation IF FITTED

    bound_min_doppler, bound_max_doppler = -10, 10
    bound_min_vmac, bound_max_vmac = 0, 15
    bound_min_rotation, bound_max_rotation = 0, 15

    if min_rv is None:
        min_rv = -5  # km/s
    if max_rv is None:
        max_rv = 5
    if min_macroturb is None:
        min_macroturb = 1
    if max_macroturb is None:
        max_macroturb = 3
    if min_rotation is None:
        min_rotation = 1
    if max_rotation is None:
        max_rotation = 3

    guess_length = 1
    if fit_vmac:
        guess_length += 1
    if fit_rotation:
        guess_length += 1

    bounds = []

    rv_guess, rv_bounds = get_simplex_guess(guess_length, min_rv, max_rv, bound_min_doppler, bound_max_doppler)
    guesses = np.array([rv_guess])
    bounds.append(rv_bounds)
    if fit_vmac:
        macro_guess, macro_bounds = get_simplex_guess(guess_length, min_macroturb, max_macroturb, bound_min_vmac, bound_max_vmac)
        guesses = np.append(guesses, [macro_guess], axis=0)
        bounds.append(macro_bounds)
    if fit_rotation:
        rotation_guess, rotation_bounds = get_simplex_guess(guess_length, min_rotation, max_rotation, bound_min_rotation, bound_max_rotation)
        guesses = np.append(guesses, [rotation_guess], axis=0)
        bounds.append(rotation_bounds)

    guesses = np.transpose(guesses)

    # check if median of the guess is 0 for either of the parameters, if so, then add a value to the guess that is
    # halfway between 0 and the max/min, depending whichever is not 0
    # otherwise, if the median is 0, the L-BFGS-B minimisation will for some reason get stuck and not move much
    if np.median(guesses[:, 0]) == 0:
        if np.abs(min_rv) > np.abs(max_rv):
            guesses[:, 0][int(np.size(guesses[:, 0]) / 2)] = np.abs(min_rv) / 2
        else:
            guesses[:, 0][int(np.size(guesses[:, 0]) / 2)] = np.abs(max_rv) / 2
    if fit_vmac:
        if np.median(guesses[:, 1]) <= 0.5:
            if np.abs(min_macroturb) > np.abs(max_macroturb):
                guesses[:, 1][int(np.size(guesses[:, 1]) / 2)] = np.abs(min_macroturb) / 2
            else:
                guesses[:, 1][int(np.size(guesses[:, 1]) / 2)] = np.abs(max_macroturb) / 2
    if fit_rotation:
        if np.median(guesses[:, -1]) <= 0.5:
            if np.abs(min_rotation) > np.abs(max_rotation):
                guesses[:, -1][int(np.size(guesses[:, -1]) / 2)] = np.abs(min_rotation) / 2
            else:
                guesses[:, -1][int(np.size(guesses[:, -1]) / 2)] = np.abs(max_rotation) / 2

    return guesses, bounds

def print_intermediate_results(intermediate_results: dict, atmosphere_type: str, night_mode: bool) -> None:
    """
    Prints intermediate results for the line
    :param intermediate_results: Dictionary with the results for the line
    :param atmosphere_type: 1D or 3D, depending on it vmic is printed or not
    :param night_mode: if True, then doesn't print anything
    :return: None
    """
    vmic_possible_columns = ["vmic", "microturb"]
    chi_sqr_possible_columns = ["chi_squared", "chi_sqr", "chisqr", "chisquared", "fitted_chisqr"]
    abundances_possible_columns = ["abundances", "abund", "abundance", "abundances", "elem_abund_dict", "abund_dict", "elem"]
    if not night_mode:
        # we want to print it, by going through each key. each key is a column name
        # 4 for everything with 7 characters total and 8 decimals for chi squared
        string_to_print = ""
        for key in intermediate_results:
            if key in chi_sqr_possible_columns:
                string_to_print += f"chi_sqr= {intermediate_results[key]:>16.8f} "
            else:
                if key.lower() in vmic_possible_columns and atmosphere_type == "3D":
                    # don't print vmic for 3D
                    pass
                else:
                    if key in abundances_possible_columns:
                        for element in intermediate_results[key]:
                            string_to_print += f"[{element}/H]= {intermediate_results[key][element]:>7.4f} "
                    else:
                        string_to_print += f"{key}= {intermediate_results[key]:>9.4f} "
        print(string_to_print)

def calc_chi_sqr_generic_lbl(feh: float, elem_abund_dict_xh: dict, vmic: float, teff: float, logg: float,
                             lmin: float, lmax: float,
                             lmin_segment: float, lmax_segment: float, temp_directory: str, line_number: int) -> float:
    """
    Calculates the chi squared for the given broadening parameters and small doppler shift for the generic lbl
    :param feh: Fe/H
    :param elem_abund_dict_xh: Abundances of the elements
    :param vmic: Microturbulence
    :param teff: Effective temperature
    :param logg: Log g
    :param ssg: Synthetic spectrum generator object
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA], where to calculate chi squared
    :param lmax: End of the line [AA], where to calculate chi squared
    :param lmin_segment: Start of the segment, where spectra is generated [AA]
    :param lmax_segment: End of the segment, where spectra is generated [AA]
    :param temp_directory: Temporary directory where code is being run
    :param line_number: Which line number/index in line_center_sorted is being fitted
    :return: best fit chi squared
    """
    macroturb = 999999    # for printing only here, in case not fitted
    rotation = 999999
    rv = 999999
    # generates spectra
    wavelength_fitted, flux_norm_fitted, flux_fitted = spectra_to_fit.configure_and_run_synthetic_code(ssg, feh, elem_abund_dict_xh, vmic,
                                                                                                       lmin_segment, lmax_segment, False,
                                                                                                       temp_dir=temp_directory, teff=teff, logg=logg)

    spectra_generated, chi_squared = check_if_spectra_generated(wavelength_fitted, spectra_to_fit.night_mode)

    param_guess, min_bounds = get_rv_macro_rotation_guess(min_macroturb=spectra_to_fit.guess_min_vmac, max_macroturb=spectra_to_fit.guess_max_vmac)
    # now for the generated abundance it tries to fit best fit macro + doppler shift.
    # Thus, macro should not be dependent on the abundance directly, hopefully
    # Seems to work way better
    function_args = (spectra_to_fit, lmin, lmax, wavelength_fitted, flux_norm_fitted)
    minimize_options = {'maxiter': 3 * 50, 'disp': False}
    res = minimize_function(calc_chi_sq_broadening, np.median(param_guess, axis=0),
                            function_args, min_bounds, 'L-BFGS-B', minimize_options)

    spectra_to_fit.rv_extra_fitted_dict[line_number] = res.x[0]
    rv = spectra_to_fit.rv_extra_fitted_dict[line_number]
    if spectra_to_fit.fit_vmac:
        spectra_to_fit.vmac_fitted_dict[line_number] = res.x[1]
        macroturb = spectra_to_fit.vmac_fitted_dict[line_number]
    else:
        macroturb = spectra_to_fit.vmac
    if spectra_to_fit.fit_rotation:
        spectra_to_fit.rotation_fitted_dict[line_number] = res.x[-1]
        rotation = spectra_to_fit.rotation_fitted_dict[line_number]
    else:
        rotation = spectra_to_fit.rotation
    chi_squared = res.fun


    intermediate_results = {"abundances": elem_abund_dict_xh, "teff": teff, "logg": logg, "rv": rv, "vmic": vmic,
                            "vmac": macroturb, "rotation": rotation, "chi_sqr": chi_squared}

    print_intermediate_results(intermediate_results, "1D", False)

    return chi_squared

if __name__ == '__main__':
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_alt_smallerldelta_ts_nlte_lesselements_hr10_2025-02-27-08-43-08.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr3_2025-03-10-10-19-24.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr10_2025-03-12-07-46-13.npz"
    #path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr13_2025-03-12-08-29-38.npz"
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_nlte_hr15n_2025-03-12-08-32-42.npz"
    payne_coeffs, wavelength_payne, labels = load_payne(path_model)
    x_min = list(payne_coeffs[-2])
    x_max = list(payne_coeffs[-1])

    wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/KPNO_FTS_flux_2960_13000_Kurucz1984.txt", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/Sun/iag_solar_flux.txt", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = np.loadtxt("./ts_spectra/sun_nlte.spec", dtype=float, unpack=True, usecols=(0, 1))
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PycharmProjects/4most/Victor/spectra_victor_jan25/G48-29", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PycharmProjects/4most/Victor/spectra_victor_jan25/G64-12", dtype=float, unpack=True)
    #data = np.loadtxt("18Sco_cont_norm.txt", dtype=float, unpack=True)
    #wavelength_obs, flux_obs = data[:, 0], data[:, 1]
    #wavelength_obs, flux_obs = np.loadtxt("ADP_18sco_snr396_HARPS_17.707g_2.norm", dtype=float, unpack=True, usecols=(0, 2), skiprows=1)
    #wavelength_obs, flux_obs = np.loadtxt("./ts_spectra/synthetic_data_sun_nlte_full.txt", dtype=float, unpack=True, usecols=(0, 1))
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PhD_2022-2025/Spectra/diff_stellar_spectra_MB/HARPS_HD122563.txt", dtype=float, unpack=True, usecols=(0, 1))
    #wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PycharmProjects/4most/Victor/spectra_victor_jan25/G64-12", dtype=float, unpack=True)

    mask = (flux_obs > 0.0) & (flux_obs < 1.2)
    wavelength_obs = wavelength_obs[mask]
    flux_obs = flux_obs[mask]

    # mask H-line cores
    h_alpha_mask = (wavelength_obs < 6562.8 - 0.5) | (wavelength_obs > 6562.8 + 0.5)
    wavelength_obs = wavelength_obs[h_alpha_mask]
    flux_obs = flux_obs[h_alpha_mask]

    resolution_val = None

    l_cut = (wavelength_obs > wavelength_payne[0]) & (wavelength_obs < wavelength_payne[-1])
    wavelength_obs = wavelength_obs[l_cut]
    flux_obs = flux_obs[l_cut]

    #p0 = [7.777, 2.94, 0.0, 1.5, -2., -2., -2., -2, 0, 3, 0]
    #p0 = [6.777, 4.54, 0.0, 1.0] + (len(labels) - 4) * [0] + [0, 0, 0]
    p0 = [7.777, 1.44, 0.0, 1.0] + (len(labels) - 4) * [0] + [0, 0, 0]

    #p0 = scale_back([0] * (len(labels)), payne_coeffs[-2], payne_coeffs[-1], label_name=None)
    # add extra 3 0s
    #p0 += [3, 3, 0]

    #def_bounds = ([3.5, 0, -4, 0.5, -3, -3, -3, -3, 0, 0, -20], [8, 5, 0.5, 3, 3, 3, 3, 3, 1e-5, 15, 20])
    def_bounds = (x_min + [0, 0, -10 + p0[-1]], x_max + [15, 15, 10 + p0[-1]])

    input_values = [None] * len(p0)
    #input_values = (6.394, 4.4297, -2.8919, 1.4783, None, None, None, None, None, 3.7822, 0, 0)
    #input_values = (4287.7906, 4.5535, -0.5972, 0.6142, None, None, None, None, None, 5.399, 0, 0)
    #input_values = (6290.449, 4.6668, -3.7677, 1.1195, None, None, None, None, None, 1.2229, 0, 0)
    #input_values[0:3] = (5777, 4.44)
    input_values[-3:] = (0, None, None)
    input_values = (None, None, None, None, 0, 0, 0, 0, 0, 0, 0)
    input_values = (None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    columns_to_pop = []
    for i, input_value in enumerate(input_values):
        if input_value is not None:
            if i == 0 and input_value > 100:
                input_value /= 1000
            p0[i] = input_value
            #def_bounds[0][i] = input_value - 1e-3
            #def_bounds[1][i] = input_value + 1e-3
            # remove that column
            columns_to_pop.append(i)

    # remove the columns from p0 and def_bounds
    for i in sorted(columns_to_pop, reverse=True):
        p0.pop(i)
        def_bounds[0].pop(i)
        def_bounds[1].pop(i)

    label_names = labels.copy()
    label_names.append('vrot')
    label_names.append('vmac')
    label_names.append('doppler_shift')

    labels_to_fit = [True] * (len(labels) + 3)
    for i, input_value in enumerate(input_values):
        if input_value is not None:
            labels_to_fit[i] = False


    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne,
        input_values,
        flux_obs,
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

    #num_samples = 10000
    #samples = np.random.multivariate_normal(mean=popt, cov=pcov, size=num_samples)

    ## 4. Make the corner plot
    ## --------------------------------------------------
    #figure = corner.corner(
    #    samples,
    #    labels=["teff", "logg", "feh", "vmic"],  # Replace with names of your parameters
    #    quantiles=[0.16, 0.5, 0.84],
    #    show_titles=True,
    #    title_fmt=".3f",
    #    title_kwargs={"fontsize": 12}
    #)

    #plt.show()

    print("Done fitting.")

    final_params = np.array(input_values).copy().astype(float)
    j = 0
    for i, input_value in enumerate(input_values):
        if input_value is None:
            final_params[i] = popt[j]
            j += 1

    #print(popt)
    labels.append('vrot')
    labels.append('vmac')
    labels.append('doppler_shift')
    j = 0
    for label, value, input_value in zip(labels, final_params, input_values):
        if input_value is None:
            std_error = np.sqrt(np.diag(pcov))[j]
            j += 1
        else:
            std_error = -1
        if label != 'teff':
            print(f"{label:<15}: {value:>10.3f} +/- {std_error:>10.3f}")
        else:
            print(f"{label:<15}: {value*1000:>10.3f} +/- {std_error*1000:>10.3f}")
    if resolution_val is not None:
        print(f"{'Resolution':<15}: {int(resolution_val):>10}")

    doppler_shift = final_params[-1]
    vmac = final_params[-2]
    vrot = final_params[-3]

    real_labels = final_params[:-3]
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