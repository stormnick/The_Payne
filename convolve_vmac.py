from __future__ import annotations
import numpy as np
from astropy import constants as const
from numpy.fft import fft, ifft
from scipy.interpolate import interp1d
import os
import pandas as pd

# Created by storm at 11.12.24


SPEED_OF_LIGHT_KMS = const.c.to('km/s').value
MIN_RD = 0.1 / SPEED_OF_LIGHT_KMS    # RESAMPLING-DISTANCE

def conv_res(wavelength, flux, resolution):
    """
    Applies convolutions to data sx, sy. Uses gaussian doppler broadedning.
    Give resolution in R for gaussian doppler broadening. Converts to doppler broadedning via v_dop = c / R
    Credits: Richard Hoppe
    """
    # convert resolution to doppler velocity
    velocity_doppler = SPEED_OF_LIGHT_KMS / resolution

    sxx = np.log(wavelength.astype(np.float64))  # original xscale in
    syy = flux.astype(np.float64)

    rd = 0.5 * np.min(np.diff(sxx))
    rd = np.max([rd, MIN_RD])

    npres = ((sxx[-1] - sxx[0]) // rd) + 1
    npresn = npres + npres % 2

    rd = (sxx[-1] - sxx[0]) / (npresn - 1)
    sxn = sxx[0] + np.arange(npresn) * rd
    syn = interp1d(sxx, syy)(sxn)

    px = (np.arange(npresn) - npresn // 2) * rd

    width = velocity_doppler / SPEED_OF_LIGHT_KMS

    py = np.exp(- (1.66511 * px / width) ** 2) / (np.sqrt(np.pi) * width)
    sxn, syn = conv_profile(sxn, syn, px, py)

    xx = np.exp(sxn)
    yy = syn

    return xx, yy


def conv_profile(xx, yy, px, py):
    norm = np.trapezoid(py, x=px)
    n = len(xx)
    dxn = (xx[-1] - xx[0]) / (n - 1)
    conv = dxn * ifft(fft(yy) * fft(np.roll(py / norm, int(n / 2))))

    return xx, np.real(conv)


def conv_macroturbulence(wavelength, flux, vmac):
    """
    Applies convolutions to data sx, sy.
    Give vmac in km/s for convolution with macroturbulence.
    Credits: Richard Hoppe
    """
    sxx = np.log(wavelength.astype(np.float64)) # original xscale in
    syy = flux.astype(np.float64)

    rd = 0.5 * np.min(np.diff(sxx))
    rd = np.max([rd, MIN_RD])

    npres = ((sxx[-1] - sxx[0]) // rd) + 1
    npresn = npres + npres%2

    rd = (sxx[-1] - sxx[0]) / (npresn-1)
    sxn = sxx[0] + np.arange(npresn) * rd
    syn = interp1d(sxx, syy)(sxn)

    px = (np.arange(npresn) - npresn // 2) * rd

    WAVE_RT_VMAC = np.arange(20) / 10.
    FLUX_RT_VMAC = [1.128, 0.939, 0.773, 0.628, 0.504, 0.399, 0.312, 0.240, 0.182, 0.133,
                    0.101, 0.070, 0.052, 0.037, 0.024, 0.017, 0.012, 0.010, 0.009, 0.007]

    WAVE_RT_VMAC = np.concatenate([-WAVE_RT_VMAC[:0:-1],WAVE_RT_VMAC])
    FLUX_RT_VMAC = np.concatenate([ FLUX_RT_VMAC[:0:-1],FLUX_RT_VMAC])
    zeta_rt1 = vmac / SPEED_OF_LIGHT_KMS * 1.433
    WAVE_RT_VMAC = WAVE_RT_VMAC * zeta_rt1
    FLUX_RT_VMAC = FLUX_RT_VMAC / zeta_rt1
    py = interp1d(WAVE_RT_VMAC, FLUX_RT_VMAC, bounds_error=False, fill_value=0)(px)
    mask = (px < WAVE_RT_VMAC[0]) + (px > WAVE_RT_VMAC[-1])
    py[mask] = 0

    sxn, syn = conv_profile(sxn, syn, px, py)

    xx = np.exp(sxn)
    yy = syn

    return xx, yy


def conv_rotation(wavelength, flux, vrot):
    """
    Applies convolutions to data sx, sy.
    Give vrot in km/s for convolution with a rotational profile.
    Credits: Richard Hoppe
    """
    beta = 1.5
    sxx  = np.log(wavelength.astype(np.float64)) # original xscale in
    syy  =        flux.astype(np.float64)

    rd = 0.5 * np.min(np.diff(sxx))
    rd = np.max([rd, MIN_RD])

    npres = ((sxx[-1] - sxx[0]) // rd) + 1
    npresn = npres + npres%2

    rd  = (sxx[-1] - sxx[0]) / (npresn-1)
    sxn = sxx[0] + np.arange(npresn) * rd
    syn = interp1d(sxx, syy)(sxn)

    px = (np.arange(npresn) - npresn // 2) * rd

    normf = SPEED_OF_LIGHT_KMS / vrot
    xi = normf*px

    xi[abs(xi) > 1] = 1

    py = (2*np.sqrt(1-xi**2) / np.pi + beta*(1-xi**2) / 2) * normf / (1 + 6/9*beta)

    sxn, syn = conv_profile(sxn, syn, px, py)

    xx = np.exp(sxn)
    yy = syn

    return xx, yy


if __name__ == '__main__':
    # temp['flxn'].shape, temp['wvl'].shape, temp['labels'].shape
    # ((num_pix, num_spectra), (num_pix,), (dim_in, num_spectra))
    # num_pix - number of output pixels in the spectrum
    # num_spectra - number of spectra for training and validation
    # dim_in - dimensionality of input parameters (e.g. teff/logg/feh/xfe)
    lmin, lmax = 5330, 5615
    extra_boundary = 10
    min_vmac, max_vmac = 4, 15
    specname_column_name = "specname"
    # teff, logg, feh, vmic, vmac, mg, ti, mn
    params_to_use = ["specname","teff","logg","feh","vmic","Mg_Fe","Ti_Fe","Mn_Fe"]

    folder_input_path = "/mnt/beegfs/gemini/groups/bergemann/users/storm/TSFitPy/synthetic_spectra_4most/Jan-10-2024-09-44-56_0.2015784826663748_NLTE_IWG7_100k_grid_labels_9-10-23_batch_0.txt"
    stellar_params_file = "spectra_parameters_noac10.csv"
    output_folder = "/mnt/beegfs/gemini/groups/bergemann/users/storm/payne/nov2024/100k_grid_nlte_batch0_hr10_vmac/"

    folder_input_path = "/mnt/beegfs/gemini/groups/bergemann/users/storm/TSFitPy/synthetic_spectra_4most/Jul-09-2024-07-11-46_0.15469816734535558_NLTE_IWG7_100k_grid_labels_9-10-23_batch_1.txt"
    stellar_params_file = "spectra_parameters_noac10.csv"
    output_folder = "/mnt/beegfs/gemini/groups/bergemann/users/storm/payne/nov2024/100k_grid_nlte_batch1_hr10_vmac/"

    all_fluxes = []

    labels = pd.read_csv(f"{folder_input_path}/{stellar_params_file}")
    labels = labels[params_to_use]
    labels['vmac'] = 0.0

    # get all files in the folder
    files = [f for f in os.listdir(folder_input_path) if f.endswith('.spec')]
    for file in files:
        print(file)
        try:
            wavelength, flux = np.loadtxt(f"{folder_input_path}/{file}", dtype=float, unpack=True, usecols=(0, 1))
        except (ValueError, OSError, FileNotFoundError) as e:
            print(f"Error in {file}: {e}")
            continue
        # cut to the lmin/lmax
        mask = (lmin - extra_boundary <= wavelength) & (wavelength <= lmax + extra_boundary)
        wavelength = wavelength[mask]
        flux = flux[mask]

        vmac = np.random.uniform(low=min_vmac, high=max_vmac)

        wavelength, flux = conv_macroturbulence(wavelength, flux, vmac)

        np.savetxt(f"{output_folder}/{file}", np.array([wavelength, flux]).T)

        # Update the vmac value in the labels dataframe
        file_index = labels[labels[specname_column_name] == file].index
        if len(file_index) > 0:  # Ensure file exists in labels
            labels.loc[file_index, 'vmac'] = vmac

    labels.to_csv(f"{output_folder}/spectra_parameters.csv", index=False)


