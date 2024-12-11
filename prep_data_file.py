from __future__ import annotations

import numpy as np
import scipy
import os

# Created by storm at 10.12.24

if __name__ == '__main__':
    # temp['flxn'].shape, temp['wvl'].shape, temp['labels'].shape
    # ((num_pix, num_spectra), (num_pix,), (dim_in, num_spectra))
    # num_pix - number of output pixels in the spectrum
    # num_spectra - number of spectra for training and validation
    # dim_in - dimensionality of input parameters (e.g. teff/logg/feh/xfe)
    lmin, lmax = 5330, 5615
    ldelta = 0.05
    specname_column_name = "specname"
    # teff, logg, feh, vmic, vmac, mg, ti, mn
    params_to_use = ["teff","logg","feh","vmic","Mg_Fe","Ti_Fe","Mn_Fe","vmac"]

    new_wavelength = np.arange(lmin, lmax+ldelta, ldelta)

    folder_input_path = "/mnt/beegfs/gemini/groups/bergemann/users/storm/TSFitPy/synthetic_spectra_4most/Jan-10-2024-09-44-56_0.2015784826663748_NLTE_IWG7_100k_grid_labels_9-10-23_batch_0.txt"
    stellar_params_file = "spectra_parameters_noac10.csv"
    folder_input_path = "/mnt/beegfs/gemini/groups/bergemann/users/storm/TSFitPy/synthetic_spectra_4most/Jul-09-2024-07-11-46_0.15469816734535558_NLTE_IWG7_100k_grid_labels_9-10-23_batch_1.txt"
    stellar_params_file = "spectra_parameters_noac10.csv"

    all_fluxes = []


    # get all files in the folder
    files = os.listdir(folder_input_path)
    for file in files:
        wavelength, flux = np.loadtxt(f"{folder_input_path}/{file}", dtype=float, unpack=True)
        # cut to the lmin/lmax
        mask = np.logical_and((lmin <= wavelength, wavelength <= lmax))
        wavelength = wavelength[mask]
        flux = flux[mask]

        # interpolate
        new_flux = np.interp(new_wavelength, wavelength, flux)
        all_fluxes.append(new_flux)
