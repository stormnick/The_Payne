from __future__ import annotations

import matplotlib.pyplot as plt
import numpy
import numpy as np
from dask.distributed import Client
from astropy.io import fits as pyfits
import pandas as pd
from tqdm import tqdm
import random
from astropy.io import fits
from convolve import conv_rotation, conv_macroturbulence, conv_res
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")



# Created by storm at 15.05.25

clight = 3e5


def gaussian1d_kernel(x, fwhm, threshold=1.e-6, **args):
    """Return a 1D-gaussian kernel
    The gaussian profile is defined as follows:

    f(x) = exp(- (x / sigma)**2)

    where sigma = FWHM / (2. * sqrt(ln(2.)))

    If sigma is a velocity, then with the above definition, the most probable velocity equals sigma
    FWHM is asked because it is the relevant quantity to compare kernels


    Parameters
    ----------
    x         : [array of float] input variable (Angstrom)
    fwhm      : [float] full width at half maximum of the gaussian function (Angstrom)
    threshold : [float, optional] threshold under which the kernel is set to zero. Default is 1e-6

    Returns
    -------
    out : [array] array filled with the 1D-gaussian kernel
    """

    factor = 2. * numpy.sqrt(
        numpy.log(2.)) / fwhm  # 1 / factor = sigma = FWHM / 1.66511; FWHM = 2. * sqrt(ln(2.)) * sigma

    kernel = numpy.exp(- x)
    kernel = numpy.exp(- (x * factor) ** 2)
    kernel[kernel <= threshold] = 0.

    return kernel


def rotation_kernel(x, fwhm, epsilon=0.6, threshold=1.e-6, **args):
    """Return the rotation kernel
    The rotation kernel is defined as follows:

    f(x) = (2. * (1. - epsilon) * sqrt(1. - (x / fwhm)^2)
           + pi / 2. * epsilon * (1 - (x / fwhm)^2)) * 1. / (pi * fwhm * (1. - epsilon / 3.))

    where epsilon is the limb-darkening coefficient
          fwhm is v * sin(i) (expressed in wavelength unit)


    Parameters
    ----------
    x         : [array of float] input variable (Angstrom)
    fwhm      : [float] full width at half maximum of the rotation function (Angstrom)
                        it is the rotation velocity in wavelength unit
    epsilon   : [float, optional] limb-darkening coefficient.  Default is 0.6
                For a wavelength-dependent limb-darkening, then you have to change epsilon at each call (e.g., epsilon = 1. - 0.3 * lambda / 5000.)
    threshold : [float, optional] threshold under which the kernel is set to zero. Default is 1e-6

    Returns
    -------
    out : [array] array filled with the rotation kernel
    """

    kernel = numpy.zeros((len(x),))

    idx = numpy.where(numpy.abs(x) < fwhm)

    kernel[idx] = (2. * (1. - epsilon) * numpy.sqrt(1. - (x[idx] / fwhm) ** 2) + numpy.pi / 2. * epsilon * (
                1. - (x[idx] / fwhm) ** 2)) / (numpy.pi * fwhm * (1. - epsilon / 3.))

    kernel[kernel <= threshold] = 0.

    return kernel


kernels = {
    "gaussian": gaussian1d_kernel,
    "rotation": rotation_kernel,
}


def _compute_kernel_at_pixel_i(x, fwhm, kernel, **args):
    """Return the value of the convolution at pixel i

    The FWHM of the convolving function (kernel) changes at each pixel (i.e. at each wavelength).

    Parameters
    ----------
    i           : [integer] pixel of the convolved function (flux) for which the convolution is computed
    x           : [array of float] independent variable (wavelength)
    y           : [array of float] dependent variable (flux)
    fwhm        : [float] full width at half maximum of the gaussian function
    half_width  : [integer] half width (in pixel) of the pixel range over which the convolution is computed
    is_velocity : [boolean]

    Returns
    -------
    out : [float] value of the convolution at pixel i

    Notes
    -----
    Arrays x and y are padded: 'half_width' have been preprended and appended. So that the pixel 'i' of the original (non-padded) array is the pixel 'i + half_width' of the padded array
    """

    computed_kernel = kernel(x, fwhm, **args)
    computed_kernel = computed_kernel / computed_kernel.sum()  # Normalised kernel
    return computed_kernel


def faltbo(wavelength, flux, fwhm, profile=None, threshold=1.e-6, limb_darkening_law=None, sanitise_borders=False,
           sanitising_length=3., sanitising_value=float("NaN")):

    if limb_darkening_law is None:
        limb_darkening_law = lambda x: 1. - 0.3 * x / 5000.

    if profile is None:
        profile = ["rotation"]

    if type(fwhm) is float or type(fwhm) is int:
        if fwhm > 0:
            fwhm = -fwhm
        fwhm = [fwhm]

    profiles = []
    for p in profile:
        if p.lower() not in kernels.keys():
            msg = _("Invalid profile")
            logger.error(msg)
            raise ValueError(msg)
        else:
            profiles.append(p.lower())

    fwhms = numpy.array(fwhm)
    if len(fwhm) == 1 and len(profiles) > 1:
        fwhms = numpy.ones((len(profiles),)) * fwhms[0]

    elif len(fwhm) > 1 and len(profiles) == 1:
        profiles = [profiles[0] for i in range(len(fwhm))]

    elif len(fwhm) != len(profiles):
        msg = _("The numbers of FWHMs and profiles are not compatible")
        logger.error(msg)
        raise ValueError(msg)

    profiles = numpy.array(profiles)

    velocity_idx = fwhms < 0.
    fwhm_idx = fwhms > 0.

    # Extend the spectrum on the left and on the right
    size = len(wavelength)

    minimum_wavelength = wavelength[0]
    maximum_wavelength = wavelength[-1]
    wavelength_left_step = wavelength[1] - minimum_wavelength
    wavelength_right_step = maximum_wavelength - wavelength[-2]

    reference_fwhm = numpy.nanmax(
        numpy.hstack([-1. * fwhms[velocity_idx] * minimum_wavelength / clight, fwhms[fwhm_idx] / 1.e3]))

    half_width = numpy.int_(50. * reference_fwhm / wavelength_left_step) + 1
    bad_pixels = numpy.int_(sanitising_length * reference_fwhm / wavelength_left_step)
    padded_size = size + 2 * half_width

    prepend_wavelength = minimum_wavelength - (numpy.linspace(half_width, 1, half_width) * wavelength_left_step)
    append_wavelength = maximum_wavelength + (numpy.arange(half_width) + 1) * wavelength_right_step
    padded_wavelength = numpy.hstack((prepend_wavelength, wavelength, append_wavelength))

    original_flux_shape = flux.shape
    flux = numpy.atleast_2d(flux)
    prepend_flux = numpy.zeros(half_width, dtype=numpy.float64)
    padded_flux = numpy.hstack((numpy.resize(prepend_flux, (flux.shape[0], half_width)), flux,
                                numpy.resize(prepend_flux, (flux.shape[0], half_width))))

    # Loop over the convolution profiles, separating those with a "width" expressed in velocity and those with a "width" expressed in wavelength
    # For each profile, loop over the pixels
    #    For each pixel,
    #       1/ update the FWHM (in Angstroms) if needed
    #       2/ compute the kernel, if needed
    #       3/ compute the convolution
    #    update the flux and apply the next convolution profile

    y = padded_flux.copy()  # where the fluxes will be updated, y is ALWAYS a padded version

    # Loop over the profile for which the width is in velocity unit
    for velocity, profile in zip(fwhms[velocity_idx], profiles[velocity_idx]):

        new_y = []

        # Loop over the pixels
        for i in numpy.arange(size):
            ibeg = 0  # Offset of the first pixel used for the convolution
            iend = 2 * half_width  # Offset of the last pixel used for the convolution

            # Wavelength array centred on 0.
            delta_x = padded_wavelength[i + ibeg:i + iend + 1] - padded_wavelength[
                i + half_width]  # Argument (lambda - lambda_0) of the shifted function (kernel) for lambda_0 = 'x[i + half_width]' and lambda runs from 'x[i + ibeg]' to 'x[i + iend + 1]'

            # Update the FWHM and convert it to wavelength
            fwhm = - velocity * padded_wavelength[
                i + half_width] / clight  # FWHM at pixel (wavelength) 'i'. Attention! x and y are padded: 'half_width' have been preprended and appended.So that the pixel 'i' of the original (non-padded) array is the pixel 'i + half_width' of the padded array

            # Limb-darkening law for the rotation profile
            if profile == "rotation":
                try:
                    epsilon = limb_darkening_law(padded_wavelength[i + half_width])
                except TypeError:
                    epsilon = limb_darkening_law
            else:
                epsilon = None

            # Compute the kernel at pixel i
            computed_kernel = _compute_kernel_at_pixel_i(delta_x, fwhm, kernels[profile], epsilon=epsilon,
                                                         threshold=threshold)

            # Compute the convolution at pixel i and save new value
            _y = y[:, i + ibeg:i + iend + 1] * computed_kernel
            new_y.append(_y.sum(axis=1))
            del _y

        del computed_kernel

        # Update y for the next convolution profile
        y = numpy.hstack((numpy.resize(prepend_flux, (flux.shape[0], half_width)), numpy.array(new_y).transpose(),
                          numpy.resize(prepend_flux, (flux.shape[0], half_width))))

    # Loop over the profile for which the width is in wavelength unit
    for fwhm, profile in zip(fwhms[fwhm_idx] / 1.e3, profiles[fwhm_idx]):  # Division by 1000 since FWHM is given in mA

        new_y = []

        # Loop over the pixels
        for i in numpy.arange(size):
            ibeg = 0  # Offset of the first pixel used for the convolution
            iend = 2 * half_width  # Offset of the last pixel used for the convolution

            # Wavelength array centred on 0.
            delta_x = padded_wavelength[i + ibeg:i + iend + 1] - padded_wavelength[
                i + half_width]  # Argument (lambda - lambda_0) of the shifted function (kernel) for lambda_0 = 'x[i + half_width]' and lambda runs from 'x[i + ibeg]' to 'x[i + iend + 1]'

            # Limb-darkening law for the rotation profile
            epsilon_is_variable = False
            if profile == "rotation":
                try:
                    epsilon = limb_darkening_law(padded_wavelength[i + half_width])
                    epsilon_is_variable = True
                except TypeError:
                    epsilon = limb_darkening_law
            else:
                epsilon = None

            # Compute the kernel at pixel i
            computed_kernel = _compute_kernel_at_pixel_i(delta_x, fwhm, kernels[profile], epsilon=epsilon,
                                                         threshold=threshold)

            # Compute the convolution at pixel i and save new value
            _y = y[:, i + ibeg:i + iend + 1] * computed_kernel
            new_y.append(_y.sum(axis=1))
            del _y

        del computed_kernel

        # Update y for the next convolution profile
        y = numpy.hstack((numpy.resize(prepend_flux, (flux.shape[0], half_width)), numpy.array(new_y).transpose(),
                          numpy.resize(prepend_flux, (flux.shape[0], half_width))))

    new_y = numpy.array(new_y).transpose()

    faltbo_bpm = numpy.zeros((new_y.shape), dtype=bool)
    faltbo_bpm[:, 0:bad_pixels] = True
    faltbo_bpm[:, -bad_pixels:] = True

    if sanitise_borders:  # Convolution is poor at borders, here it is fixed
        new_y[:, 0:bad_pixels] = sanitising_value
        new_y[:, -bad_pixels:] = sanitising_value

    return wavelength, new_y.reshape(original_flux_shape)



if __name__ == '__main__':

    wavelength, flux = np.loadtxt("../ts_spectra/synthetic_data_sun_nlte.txt", unpack=True, usecols=(0, 1), dtype=float, skiprows=1)
    wavelength, flux = np.loadtxt("/Users/storm/PycharmProjects/4most/4mostification_codes/test2/part1/2163.spec", unpack=True, usecols=(0, 1), dtype=float, skiprows=1)

    vsini = 0.28
    resolution = 20000

    import time

    wavelength_1, flux_1 = conv_rotation(wavelength, flux, vsini)
    wavelength_1, flux_1 = conv_res(wavelength_1, flux_1, resolution)

    time_start = time.perf_counter()
    wavelength_faltbo, flux_faltbo = faltbo(wavelength, flux, vsini)
    print(f"time {time.perf_counter() - time_start:.2f} seconds")
    time_start = time.perf_counter()
    wavelength_fft, flux_fft = conv_rotation(wavelength, flux, vsini)
    print(f"time {time.perf_counter() - time_start:.2f} seconds")

    plt.figure(figsize=(14, 7))
    plt.scatter(wavelength_faltbo, flux_faltbo, label="FALTBO", s=3)
    plt.scatter(wavelength_fft, flux_fft, label="FFT", s=3)
    plt.legend()
    plt.xlim(6140.75, 6142.75)
    plt.show()

    flux_interp = interp1d(
        wavelength_faltbo,
        flux_faltbo,
        kind='linear',
        bounds_error=False,
        fill_value=1
    )

    plt.figure(figsize=(14, 7))
    plt.plot(wavelength_fft, flux_interp(wavelength_fft) - flux_fft, label="FALTBO")
    plt.legend()
    plt.xlim(6140.75, 6142.75)
    plt.show()