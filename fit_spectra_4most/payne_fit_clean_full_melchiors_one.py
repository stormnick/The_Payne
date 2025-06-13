from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy
from The_Payne import spectral_model
from convolve import *
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from payne_fit_clean_full import fit_teff, fit_logg, fit_feh, fit_one_xfe_element, process_spectra, load_payne

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 03.03.25

if __name__ == '__main__':
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_2025-03-27-08-06-34.npz"

    payne_coeffs, wavelength_payne, labels = load_payne(path_model)
    x_min = list(payne_coeffs[-2])
    x_max = list(payne_coeffs[-1])

    label_names = labels.copy()
    label_names.append('vsini')
    label_names.append('vmac')
    label_names.append('doppler_shift')

    stellar_rv = 0
    resolution_val = None

    # get all filenames in the folder
    folder = "/Users/storm/Downloads/Cont/"
    folder_spectra = "/Users/storm/Downloads/Science/"
    files = os.listdir(folder)

    fitted_values = pd.DataFrame(columns=["spectraname", *labels, "vsini", "vmac", "doppler_shift"])

    file_to_fit = "HD210027.txt"

    for file in files:
        if file != file_to_fit:
            continue

        print(f"Fitting {file}")
        continuum = np.load(f"{folder}{file}")
        spectra = np.load(f"{folder_spectra}{file}")
        wavelength_obs = continuum[0]
        flux_obs = spectra[1] / continuum[1]
        np.savetxt(file, np.array([wavelength_obs, flux_obs]).T)
        stellar_rv = 0

        start_time = time.perf_counter()

        h_line_cores = pd.read_csv("../linemasks/h_cores.csv")
        h_line_cores = list(h_line_cores['ll'])

        wavelength_obs, flux_obs = process_spectra(wavelength_payne, wavelength_obs, flux_obs, h_line_cores, h_line_core_mask_dlam=0.5)

        final_parameters = {}
        final_parameters_std = {}

        # 1. TEFF
        # fits teff, logg, feh, vmac, rv for h-alpha lines
        fe_lines = pd.read_csv("../fe_lines_hr_good.csv")
        fe_lines = list(fe_lines["ll"])
        h_line_cores += fe_lines

        teff, teff_std = fit_teff(labels, payne_coeffs, x_min, x_max, stellar_rv, h_line_cores, wavelength_obs,
                                  flux_obs, wavelength_payne, resolution_val, silent=False)

        final_parameters["teff"] = teff
        final_parameters_std["teff"] = teff_std

        # 2. LOGG, FEH, VMAC, RV, also fit Mg, Ca

        logg, logg_std, doppler_shift, doppler_shift_std = fit_logg(final_parameters, labels, payne_coeffs, x_min,
                                                                    x_max, stellar_rv, wavelength_obs, flux_obs,
                                                                    wavelength_payne, resolution_val, silent=False)

        final_parameters["logg"] = logg
        final_parameters["doppler_shift"] = doppler_shift
        final_parameters_std["logg"] = logg_std
        final_parameters_std["doppler_shift"] = doppler_shift_std

        # 3. FEH, VMIC, VMAC
        feh, feh_std, vmic, vmic_std, vsini, vsini_std, vmac, vmac_std = fit_feh(final_parameters, labels, payne_coeffs,
                                                                                 x_min, x_max, stellar_rv,
                                                                                 wavelength_obs,
                                                                                 flux_obs, wavelength_payne,
                                                                                 resolution_val,
                                                                                 silent=False, fit_vsini=True,
                                                                                 fit_vmac=False)

        final_parameters["feh"] = feh
        final_parameters["vmic"] = vmic
        final_parameters["vsini"] = vsini
        final_parameters["vmac"] = vmac
        final_parameters_std["feh"] = feh_std
        final_parameters_std["vmic"] = vmic_std
        final_parameters_std["vsini"] = vsini_std
        final_parameters_std["vmac"] = vmac_std

        # 4. REMAINING ELEMENTS ONE-BY-ONE
        # find how many _Fe labels are there
        elements_to_fit = []
        for i, label in enumerate(labels):
            if label.endswith("_Fe"):
                elements_to_fit.append(label)

        for element_to_fit in elements_to_fit:
            xfe, xfe_std = fit_one_xfe_element(final_parameters, element_to_fit, labels, payne_coeffs, x_min, x_max,
                                               stellar_rv, wavelength_obs, flux_obs,
                                               wavelength_payne, resolution_val, silent=True)

            final_parameters[element_to_fit] = xfe
            final_parameters_std[element_to_fit] = xfe_std

        print(f"Fitted {file} in {time.perf_counter() - start_time:.2f} seconds")

        # PRINT RESULTS
        for label in label_names:
            value = final_parameters[label]
            std_error = final_parameters_std[label]
            if label != 'teff':
                print(f"{label:<15}: {value:>10.3f} +/- {std_error:>10.3f}")
            else:
                print(f"{label:<15}: {value * 1000:>10.3f} +/- {std_error * 1000:>10.3f}")
        if resolution_val is not None:
            print(f"{'Resolution':<15}: {int(resolution_val):>10}")

        doppler_shift = final_parameters['doppler_shift']
        vmac = final_parameters['vmac']
        vsini = final_parameters['vsini']

        final_params = []

        for label in labels:
            final_params.append(final_parameters[label])

        real_labels = final_params
        scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
        payne_fitted_spectra = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels,
                                                                           NN_coeffs=payne_coeffs, kovalev_alt=True)

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


        real_labels = final_params
        real_labels[0:3] = [3.904,1.06,-0.26]
        real_labels[7] = -0.7
        scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
        payne_fitted_spectra2 = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels,
                                                                           NN_coeffs=payne_coeffs, kovalev_alt=True)

        wavelength_payne_plot2 = wavelength_payne
        if vmac > 1e-3:
            wavelength_payne_plot2, payne_fitted_spectra2 = conv_macroturbulence(wavelength_payne_plot2,
                                                                               payne_fitted_spectra2, vmac)
        if vsini > 1e-3:
            wavelength_payne_plot2, payne_fitted_spectra2 = conv_rotation(wavelength_payne_plot2, payne_fitted_spectra2,
                                                                        vsini)
        if resolution_val is not None:
            wavelength_payne_plot2, payne_fitted_spectra2 = conv_res(wavelength_payne_plot2, payne_fitted_spectra2,
                                                                   resolution_val)

        plt.figure(figsize=(18, 6))
        plt.scatter(wavelength_obs, flux_obs, label="Observed", s=3, color='k')
        plt.plot(wavelength_payne_plot * (1 + (doppler_shift / 299792.)), payne_fitted_spectra, label="Payne",
                 color='r')
        plt.plot(wavelength_payne_plot2 * (1 + (doppler_shift / 299792.)), payne_fitted_spectra2, label="Payne test", color='b')
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
        plt.show()

