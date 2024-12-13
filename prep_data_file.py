from __future__ import annotations

import numpy as np
import os
import pandas as pd
from tqdm import tqdm

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
    params_to_use = ["specname","teff","logg","feh","vmic","vmac","Mg_Fe","Ti_Fe","Mn_Fe"]
    params_to_save = ["teff","logg","feh","vmic","vmac","Mg_Fe","Ti_Fe","Mn_Fe"]

    new_wavelength = np.arange(lmin, lmax+ldelta, ldelta)

    folder_input_paths = ["/mnt/beegfs/gemini/groups/bergemann/users/storm/payne/nov2024/100k_grid_nlte_batch0_hr10_vmac",
                          "/mnt/beegfs/gemini/groups/bergemann/users/storm/payne/nov2024/100k_grid_nlte_batch1_hr10_vmac"]
    stellar_params_file = "spectra_parameters.csv"
    file_output_path = "/mnt/beegfs/gemini/groups/bergemann/users/storm/payne/nov2024/100k_grid_nlte_batch01_hr10_vmac_dec2024"

    all_fluxes = []
    labels = []


    for folder_input_path in folder_input_paths:
        labels_pd = pd.read_csv(f"{folder_input_path}/{stellar_params_file}")
        labels_pd = labels_pd[params_to_use]
        labels_pd['teff'] = labels_pd['teff'] / 1000
        # get all files in the folder
        files = [f for f in os.listdir(folder_input_path) if f.endswith('.spec')]
        for file in tqdm(files):
            try:
                wavelength, flux = np.loadtxt(f"{folder_input_path}/{file}", dtype=float, unpack=True, usecols=(0, 1))
            except (ValueError, OSError, FileNotFoundError) as e:
                print(f"Error in {file}: {e}")
                continue
            # cut to the lmin/lmax
            mask = (lmin <= wavelength) & (wavelength <= lmax)
            wavelength = wavelength[mask]
            flux = flux[mask]

            # check if flux >= 0
            if np.any(flux < 0) or np.any(np.isnan(flux)):
                print(f"Error in {file}: Negative flux values or NaNs")
                continue

            # find how many times the file is in the labels
            idx_count = len(labels_pd[labels_pd[specname_column_name] == file])
            if idx_count == 1:
                # interpolate
                new_flux = np.interp(new_wavelength, wavelength, flux)
                all_fluxes.append(new_flux)
                labels.append(labels_pd[labels_pd[specname_column_name] == file][params_to_save].values[0])
            else:
                print(f"Error in {file}: {idx_count} entries in the labels")

    # need to save .npz in the following format:
    # temp['flx'].shape, temp['wvl'].shape, temp['labels'].shape
    # ((num_pix, num_spectra), (num_pix,), (dim_in, num_spectra))
    num_pix = len(new_wavelength)
    num_spectra = len(all_fluxes)
    dim_in = len(params_to_save)
    flx = np.array(all_fluxes).T
    wvl = new_wavelength
    labels = np.array(labels).T

    np.savez(f"{file_output_path}.npz", flxn=flx, wvl=wvl, labels=labels)


