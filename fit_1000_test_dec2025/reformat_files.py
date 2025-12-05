from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from astropy.io import fits

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 03.12.25


if __name__ == '__main__':
    file_path = "/Users/storm/Downloads/1000_mixed_synthetic_spectra/20221110/"
    files = ["20221110_HB11_20251102.fits", "20221110_HG11_20251102.fits", "20221110_HR11_20251102.fits"]

    output_path = "/Users/storm/Downloads/1000_mixed_synthetic_spectra/20221110/spectra"

    for file in files:
        # extract if it is HB, HG, or HR
        fiber_type = file.split("_")[1][1:2]  # HB, HG, HR
        print(f"Processing file: {file} of type {fiber_type}")

        hdul = fits.open(file_path + file)
        obmetab = hdul[1].data  # OBMETATAB
        spec_tab = hdul[2].data  # SPECTAB
        fib_tab = hdul[3].data  # FIBMETATAB
        prov = hdul[4].data     # PROVTAB

        n_spec = len(spec_tab)
        n_pix = len(spec_tab['FLUX'][0])  # 13761

        # Wavelength grid per spectrum (allowing it to vary per fibre if needed)
        waves = np.empty((n_spec, n_pix), dtype=float)

        for i in range(n_spec):
            lam0 = fib_tab['WAVELMIN'][i]
            dlam = fib_tab['SPEC_BIN'][i]
            waves[i, :] = lam0 + np.arange(n_pix) * dlam

        waves = waves * 10

        file_number = len(spec_tab['SPECUID'])
        #print(f"Number of spectra in file: {file_number}")
        #print(prov['SPECUID'])
        #print(spec_tab.columns)
        #continue

        for i in range(file_number):
            specuid = spec_tab['SPECUID'][i]
            specuid = specuid[:-6] + "3" + specuid[-5:]
            wavelength = waves[i, :]
            flux = spec_tab['FLUX'][i, :]

            output_filename = f"{output_path}/spectrum_{specuid}_{fiber_type}.txt"
            np.savetxt(output_filename, np.column_stack((wavelength, flux)), header="Wavelength[A]   Flux")
            print(f"Saved spectrum to {output_filename}")
