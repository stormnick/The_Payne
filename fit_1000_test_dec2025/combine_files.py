from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 03.12.25

if __name__ == '__main__':

    output_path = "/Users/storm/Downloads/1000_mixed_synthetic_spectra/20221110/combined"
    input_path = "/Users/storm/Downloads/1000_mixed_synthetic_spectra/20221110/spectra"

    # get all files in the input_path
    files = [f for f in os.listdir(input_path) if f.endswith('_R.txt')]

    # load all files and combine them
    for file in files:
        wavelength_red, flux_red = np.loadtxt(os.path.join(input_path, file), unpack=True)
        try:
            wavelength_blue, flux_blue = np.loadtxt(os.path.join(input_path, file.replace('_R.txt', '_B.txt')), unpack=True)
            wavelength_green, flux_green = np.loadtxt(os.path.join(input_path, file.replace('_R.txt', '_G.txt')), unpack=True)

            # combine all three
            wavelength_combined = np.concatenate((wavelength_blue, wavelength_green, wavelength_red))
            flux_combined = np.concatenate((flux_blue, flux_green, flux_red))
            # sort by wavelength
            sorted_indices = np.argsort(wavelength_combined)
            wavelength_combined = wavelength_combined[sorted_indices]
            flux_combined = flux_combined[sorted_indices]

            # save to output_path
            output_filename = os.path.join(output_path, file.replace('_R.txt', '_combined.txt'))
            np.savetxt(output_filename, np.column_stack((wavelength_combined, flux_combined)), header="Wavelength[A]   Flux")
            print(f"Saved combined spectrum to {output_filename}")
        except FileNotFoundError:
            print(f"Missing blue or green file for {file}, skipping.")

