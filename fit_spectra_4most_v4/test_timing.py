from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from payne_fit_clean_full import (fit_teff, fit_logg, fit_feh, fit_one_xfe_element, process_spectra, load_payne,
                                  fit_teff_logg, plot_fitted_payne)
import datetime
import time
import spectral_model

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 13.06.25


if __name__ == '__main__':
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_batch01_medium_test2training_reducedlogg_2025-06-11-08-02-05.npz"
    #path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_batch01_medium_test2training_reducedlogg_altarch_2025-06-12-10-27-38.npz"

    payne_coeffs, wavelength_payne, labels = load_payne(path_model)
    x_min = list(payne_coeffs[-2])
    x_max = list(payne_coeffs[-1])

    real_labels = [5.777, 4.44, 0, 1] + [0] * (len(labels) - 4)  # Example labels for testing

    start_time = time.perf_counter()

    for i in range(500):
        scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
        spec_payne = spectral_model.get_spectrum_from_neural_net(
            scaled_labels=scaled_labels,
            NN_coeffs=payne_coeffs,
            pixel_limits=None
        )

    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")