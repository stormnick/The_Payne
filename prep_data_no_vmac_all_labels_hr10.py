from __future__ import annotations

import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from add_vmac import conv_res
import datetime, time

# Created by storm at 10.12.24

if __name__ == '__main__':
    # temp['flxn'].shape, temp['wvl'].shape, temp['labels'].shape
    # ((num_pix, num_spectra), (num_pix,), (dim_in, num_spectra))
    # num_pix - number of output pixels in the spectrum
    # num_spectra - number of spectra for training and validation
    # dim_in - dimensionality of input parameters (e.g. teff/logg/feh/xfe)
    lmin, lmax = 5330, 5619
    ldelta = 0.02
    specname_column_name = "specname"
    # Li_Fe,C_Fe,N_Fe,O_Fe,Na_Fe,Mg_Fe,Al_Fe,Si_Fe,Ca_Fe,V_Fe,Ti_Fe,Cr_Fe,Mn_Fe,Co_Fe,Ni_Fe,Sr_Fe,Y_Fe,Zr_Fe,Ba_Fe,Ce_Fe,Eu_Fe
    params_to_use = [
    "specname", "teff", "logg", "feh", "vmic", "C_Fe", "Mg_Fe", "Ca_Fe", "Ti_Fe", "Y_Fe", "Mn_Fe"
]
    params_to_save = [
    "teff", "logg", "feh", "vmic", "C_Fe", "Mg_Fe", "Ca_Fe", "Ti_Fe", "Y_Fe", "Mn_Fe"
]

    new_wavelength = np.arange(lmin, lmax + ldelta / 10, ldelta)

    # print current time
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    folder_input_paths = [
        "/mnt/beegfs/gemini/groups/bergemann/users/storm/TSFitPy/synthetic_spectra_4most/2025-03-04-07-46-08_0.08554153559308308_NLTE_grid_to_generate_batch0_hr10_3mar2025.txt/"
    ]
    stellar_params_file = "spectra_parameters.csv"
    file_output_path = (
        "./grid_nlte_v2_ts_batch0_hr10_novmac_inf_res_mar2025"
    )
    all_fluxes = []
    labels = []

    print(f"Reading files from {folder_input_paths} and saving to {file_output_path}.npz")
    print(f"lmin: {lmin}, lmax: {lmax}, ldelta: {ldelta}")

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
            if np.any(flux <= 1e-8) or np.any(np.isnan(flux)):
                print(f"Error in {file}: Negative flux values or NaNs")
                continue

            #wavelength, flux = conv_res(wavelength, flux, 1_000_000)

            # set any flux > 1 to 1
            flux[flux > 1] = 1

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
    label_names = params_to_save

    np.savez(f"{file_output_path}.npz", flxn=flx, wvl=wvl, labels=labels, label_names=label_names)
    print(f"Saved to {file_output_path}.npz")
    print(f"flxn.shape: {flx.shape}, wvl.shape: {wvl.shape}, labels.shape: {labels.shape}, label_names: {label_names}")
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

