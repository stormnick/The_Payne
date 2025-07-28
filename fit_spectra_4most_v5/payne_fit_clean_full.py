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
from dataclasses import dataclass, field, replace

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 03.03.25

@dataclass(slots=True)
class PayneParams:
    labels: list[str]
    payne_coeffs: tuple
    wavelength_payne: np.ndarray
    x_min: list
    x_max: list
    resolution_val: float | None = None

@dataclass(slots=True)
class Parameter:
    label_name: str
    value: float
    std: float = -99            # default: not set
    fit: bool = True            # default: include in optimisation
    min_value: float | None = None
    max_value: float | None = None

    def bounds(self) -> tuple[float, float]:
        return self.min_value, self.max_value

    def fmt(self) -> str:
        flag = "fit " if self.fit else "fix "
        return f"{flag} {self.value:10.3f} ± {self.std:7.3f}"

@dataclass(slots=True)
class LSQSetup:
    p0: list[float]
    input_values: list[float | None]
    bounds: tuple[list[float], list[float]]
    labels: list[str]               # labels actually being fitted, same order as p0

@dataclass(slots=True)
class StellarParameters:
    teff: Parameter
    logg: Parameter
    feh:  Parameter
    vmic: Parameter
    vsini: Parameter
    vmac: Parameter
    doppler_shift: Parameter
    abundances: dict[str, Parameter] = field(default_factory=dict)

    _core_map = {
        "teff": "teff",  # label → attribute name
        "logg": "logg",
        "feh": "feh",
        "vmic": "vmic",
        "vsini": "vsini",
        "vmac": "vmac",
        "doppler_shift": "doppler_shift",
    }

    def __str__(self) -> str:
        # Column widths
        name_w, val_w = 15, 20
        header = f"{'Label':{name_w}} | {'Mode & Value':{val_w}}\n" + "-" * (name_w + val_w + 3)

        rows: list[str] = []

        # core labels in fixed order
        for lab in ("teff", "logg", "feh", "vmic", "vsini", "vmac", "doppler_shift"):
            param: Parameter = getattr(self, lab)
            rows.append(f"{lab:{name_w}} | {param.fmt()}")

        # abundances sorted alphabetically
        for lab, param in sorted(self.abundances.items()):
            rows.append(f"{lab:{name_w}} | {param.fmt()}")

        return "\n".join([header, *rows])

    def build_lsq_inputs(
        self,
        payne_labels,
        labels_to_fit
    ) -> LSQSetup:
        """
        Assemble the vectors required by `scipy.optimize.curve_fit`.

        Parameters
        ----------
        payne_labels
            Full label list used by The Payne *excluding* vsini/vmac/doppler_shift.
        labels_to_fit
            Iterable of label names that should be optimised.

        Returns
        -------
        LSQSetup
            p0, input vector (with None placeholders), per-parameter bounds,
            and the list of labels whose order matches p0.
        """
        # (1) final label sequence ------------------------------------------------
        labels = list(payne_labels) + ["vsini", "vmac", "doppler_shift"]

        # (2) fast lookup helpers -------------------------------------------------
        fit_set: set[str] = set(labels_to_fit)
        p0: list[float] = []
        pmin: list[float] = []
        pmax: list[float] = []
        input_values: list[float | None] = []
        fitted_labels: list[str] = []

        # (3) build vectors -------------------------------------------------------
        for lab in labels:
            # select the Parameter object
            if lab in self._core_map:
                param: Parameter = getattr(self, self._core_map[lab])
            else:
                # safe even if abundances is regular dict: key order comes from labels[]
                param = self.abundances.get(lab)
                if param is None:
                    raise KeyError(f"StellarParameters missing abundance '{lab}'")

            # decide whether we fit this label
            is_fitted = param.fit and lab in fit_set
            input_values.append(None if is_fitted else param.value)

            if is_fitted:
                p0.append(param.value)
                lo, hi = param.bounds()
                pmin.append(lo if lo is not None else -np.inf)
                pmax.append(hi if hi is not None else  np.inf)
                fitted_labels.append(lab)

        return LSQSetup(
            p0=p0,
            input_values=input_values,
            bounds=(pmin, pmax),
            labels=fitted_labels,
        )

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


def create_default_stellar_parameters(payne_parameters: PayneParams):
    p0 = scale_back([0] * (len(payne_parameters.labels)), payne_parameters.x_min, payne_parameters.x_max)

    p0[4:] = (len(p0) - 4) * [0.0]  # set all other parameters to 0, except the first four

    stellar_parameters = StellarParameters(
        teff=Parameter(value=p0[0], std=-99, fit=True, min_value=payne_parameters.x_min[0], max_value=payne_parameters.x_max[0], label_name="teff"),
        logg=Parameter(value=p0[1], std=-99, fit=True, min_value=payne_parameters.x_min[1], max_value=payne_parameters.x_max[1], label_name="logg"),
        feh=Parameter(value=p0[2], std=-99, fit=True, min_value=payne_parameters.x_min[2], max_value=payne_parameters.x_max[2], label_name="feh"),
        vmic=Parameter(value=p0[3], std=-99, fit=True, min_value=payne_parameters.x_min[3], max_value=payne_parameters.x_max[3], label_name="vmic"),
        vsini=Parameter(value=3, std=-99, fit=True, min_value=0.0, max_value=100.0, label_name="vsini"),
        vmac=Parameter(value=0, std=-99, fit=False, min_value=0.0, max_value=100.0, label_name="vmac"),
        doppler_shift=Parameter(value=0, std=-99, fit=True, min_value=-5.0, max_value=5.0, label_name="doppler_shift"),
    )

    # add abundances
    for i, label in enumerate(payne_parameters.labels[4:]):
        stellar_parameters.abundances[label] = Parameter(
            value=0.0, std=-99, fit=True, min_value=payne_parameters.x_min[i + 4], max_value=payne_parameters.x_max[i + 4], label_name=label
        )

    return stellar_parameters


def fit_stellar_parameters(stellar_parameters: StellarParameters, payne_parameters: PayneParams, wavelength_obs, flux_obs, do_hydrogen_lines=False, silent=False):
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
    lines_to_use = mg_line_cut + ca_line_cut + fe_line_cut
    lines_to_cut = mg_line_payne_cut + ca_line_payne_cut + fe_line_payne_cut

    if do_hydrogen_lines:
        lines_to_use += h_line_cut
        lines_to_cut += h_line_payne_cut
        logg_lines += list(h_line_cores)

    lsq = stellar_parameters.build_lsq_inputs(
        payne_parameters.labels,
        ["teff", "logg", "feh", "vmic", "vsini", "vmac", "Mg_Fe", "Ca_Fe", "doppler_shift"]
    )

    wavelength_obs_cut_to_lines, flux_obs_cut_to_lines, wavelength_payne_cut, combined_mask_payne = cut_to_just_lines(
        wavelength_obs, flux_obs, wavelength_payne, logg_lines, stellar_rv, obs_cut_aa=lines_to_use,
        payne_cut_aa=lines_to_cut)
    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne_cut,
        lsq.input_values,
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
        p0=lsq.p0,
        bounds=lsq.bounds,
        max_nfev=10e5
    )

    if not silent:
        print(f"Done fitting in {time.perf_counter() - time_start:.2f} seconds")
        for i, label in enumerate(lsq.labels):
            print(f"Fitted {label:>16}: {popt[i]:>8.3f} +/- {np.sqrt(np.diag(pcov))[i]:>8.3f}")

    sigmas = np.sqrt(np.diag(pcov))  # vector of 1-sigma errors

    for lab, val, err in zip(lsq.labels, popt, sigmas):
        if lab in stellar_parameters._core_map:  # stellar parameters
            attr = stellar_parameters._core_map[lab]
            old_param = getattr(stellar_parameters, attr)
        else:  # abundance
            old_param = stellar_parameters.abundances[lab]

        old_param.value, old_param.std = float(val), float(err)

    return stellar_parameters

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

def fit_one_xfe_element(element_to_fit: str, stellar_parameters: StellarParameters, payne_parameters: PayneParams, wavelength_obs, flux_obs, silent=False):
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

    dlam = scale_dlam(dlam, stellar_parameters.vsini.value + stellar_parameters.vmac.value)

    lsq = stellar_parameters.build_lsq_inputs(payne_parameters.labels, [element_to_fit])

    wavelength_obs_cut_to_lines, flux_obs_cut_to_lines, wavelength_payne_cut, combined_mask_payne = cut_to_just_lines(
        wavelength_obs, flux_obs, wavelength_payne, element_lines, stellar_rv, obs_cut_aa=list(dlam), payne_cut_aa=list(np.asarray(dlam) * 1.5))

    model_func = make_model_spectrum_for_curve_fit(
        payne_coeffs,
        wavelength_payne_cut,
        lsq.input_values,
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
            p0=lsq.p0,
            bounds=lsq.bounds,
        )
        fitted_value = float(popt[0])
        fitted_error = float(np.sqrt(np.diag(pcov))[0])
    except ValueError:
        print(f"Fitting failed for element {element_to_fit}")
        fitted_value = -99
        fitted_error = -99

    if not silent:
        print(f"Done fitting in {time.perf_counter() - time_start:.2f} seconds")
        print(f"Fitted {element_to_fit}: {fitted_value:.3f} +/- {fitted_error:.3f}")

    if element_to_fit in stellar_parameters.abundances:
        stellar_parameters.abundances[element_to_fit].value = fitted_value
        stellar_parameters.abundances[element_to_fit].std = fitted_error

    return stellar_parameters


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
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_batch01_medium_test2training_reducedlogg_altarch_2025-06-16-06-28-26.npz"
    payne_coeffs, wavelength_payne, labels = load_payne(path_model)
    x_min = list(payne_coeffs[-2])
    x_max = list(payne_coeffs[-1])

    label_names = labels.copy()
    label_names.append('vsini')
    label_names.append('vmac')
    label_names.append('doppler_shift')

    resolution_val = None

    wavelength_obs, flux_obs = np.loadtxt("/Users/storm/PycharmProjects/payne/observed_spectra_to_test/Sun_melchiors_spectrum.txt", dtype=float, unpack=True)
    stellar_rv = 0

    h_line_cores = pd.read_csv("../linemasks/h_cores.csv")
    h_line_cores = list(h_line_cores['ll'])

    wavelength_obs, flux_obs = process_spectra(wavelength_payne, wavelength_obs, flux_obs, h_line_cores, h_line_core_mask_dlam=0.5)

    payne_parameters = PayneParams(
        payne_coeffs=payne_coeffs,
        wavelength_payne=wavelength_payne,
        labels=labels,
        x_min=x_min,
        x_max=x_max,
        resolution_val=resolution_val
    )

    stellar_parameters = create_default_stellar_parameters(payne_parameters)
    stellar_parameters = fit_stellar_parameters(stellar_parameters, payne_parameters, wavelength_obs, flux_obs, silent=False)

    elements_to_fit = []
    for i, label in enumerate(labels):
        if label.endswith("_Fe") or label == "A_Li":
            elements_to_fit.append(label)

    for element_to_fit in elements_to_fit:
        stellar_parameters = fit_one_xfe_element(element_to_fit, stellar_parameters, payne_parameters, wavelength_obs, flux_obs, silent=True)

    print(stellar_parameters)

    final_params = stellar_parameters.build_lsq_inputs(labels, []).input_values
    vmac = final_params[-2]
    vsini = final_params[-3]
    doppler_shift = final_params[-1]

    real_labels = final_params[:-3]  # take all but the last three parameters
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
    plt.ylim(0.0, 1.05)
    plt.xlim(wavelength_payne_plot[0], wavelength_payne_plot[-1])
    plt.show()