from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import glob
import os
from convolve import conv_res
from astropy.io import fits

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 03.06.25


input_dir = "/Users/storm/PhD_2025/02.22 Payne/real_spectra_to_fit/archive_Ruchti2013/"
output_dir = "/Users/storm/PhD_2025/02.22 Payne/real_spectra_to_fit/converted/"

# get all files
files = glob.glob(f"{input_dir}/*")

#=====================================================================
def tbdfits(file, ext=None, survey=None, verbose=True):
    """
    Reads a FITS file and extracts wavelength and flux data.

    Parameters:
        file (str): Path to the FITS file.
        ext (int): Extension number in the FITS file.
        survey (str): Survey type ("apo", "ges", "harps").

    Returns:
        tuple: (wavel, flux) where
            - wavel is a NumPy array of wavelengths
            - flux is a NumPy array of flux values
    """
    if ext is None or survey is None:
        print("Usage: ext: 0 ... 6, survey: apo, ges, harps")
        return

    # Read the FITS file
    with fits.open(file) as hdul:
        header = hdul[ext].header
        flux = hdul[ext].data

    # Retrieve the wavelength zero point (CRVAL1)
    cr = header.get('CRVAL1')
    if cr is None:
        raise ValueError("CRVAL1 not found in the header.")

    if verbose: print("CRVAL1 (wavelength zero point):", cr)

    # Determine the wavelength increment based on the survey
    if survey == "ges":
        cdelt = header.get('CD1_1')
    elif survey in ["apo", "harps"]:
        cdelt = header.get('CDELT1')
    else:
        raise ValueError(f"Unknown survey type: {survey}")

    if cdelt is None:
        raise ValueError("CDELT1/CD1_1 not found in the header.")

    if verbose: print("Wavelength increment (CDELT1/CD1_1):", cdelt)

    # Check for SNR in the header
    snr_keys = [key for key in header if key.startswith("SNR")]
    if len(snr_keys) == 1:
        snr = header[snr_keys[0]]
        if verbose: print("SNR:", snr)

    lam = np.arange(len(flux)) * cdelt + cr
    if survey == "apo":
        lam = 10 ** lam

    return lam, flux, header

bad_data = np.loadtxt("ruchti_delete.txt", dtype=str)

for file in files:
    if any(bad in file for bad in bad_data):
        print(f"Skipping {file} as it is in the bad data list.")
        continue
    print(f"Converting {file}")
    wavelength_obs, flux_obs, _ = tbdfits(file, ext=0, survey='ges', verbose=False)
    # Assuming the first extension contains the data

    # remove any with flux = 0
    mask = flux_obs > 0
    wavelength_obs = wavelength_obs[mask]
    flux_obs = flux_obs[mask]
    # remove any with flux > 1.2
    mask = flux_obs < 1.2
    wavelength_obs = wavelength_obs[mask]
    flux_obs = flux_obs[mask]
    filename = os.path.basename(file)

    #plt.plot(wavelength_obs, flux_obs, label=filename)
    #plt.legend()
    #plt.show()

    #wavelength_obs, flux_obs = np.loadtxt(file, usecols=(0, 1), unpack=True, dtype=float)

    resolution_val = 22100

    wavelength_obs, flux_obs = conv_res(wavelength_obs, flux_obs, resolution_val)

    output_file = f"{output_dir}{filename}".replace(".fits", ".txt")
    np.savetxt(output_file, np.array([wavelength_obs, flux_obs]).T, fmt="%.5f")

