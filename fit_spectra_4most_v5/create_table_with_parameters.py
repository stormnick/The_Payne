from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import fit_systematic_error

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 29.08.25

if __name__ == '__main__':
    data = pd.read_csv("merged_data_with_tsfitpy_new_v2.csv")

    # print values of the column Sr_Fe_tsfitpy
    print(data['Sr_Fe_tsfitpy'])

    elements = data.columns.tolist()
    # only take those ending with _Fe
    elements = [el for el in elements if el.endswith("_Fe")]
    elements.append("A_Li")
    elements.remove('e_Mg_Fe')
    print(elements)
    # join elements using " & "
    string = ""
    for el in elements:
        if el != "A_Li":
            string += f'[{el.replace("_", r"/")}]' + " & "
        else:
            string += "A(Li)"

    elements1 = elements[:3]
    elements2 = elements[3:11]
    elements3 = elements[11:]

    print(string)

    instruments = []

    all_spectra_name = []
    for row in data.iterrows():
        spectraname = row[1]['spectraname']
        instrument = ""
        if "NARVAL" in spectraname:
            instrument = "NARVAL"
            spectraname = spectraname.replace("NARVAL_", "").replace(".txt", "")
        elif "HARPS" in spectraname:
            instrument = "HARPS"
            spectraname = spectraname.replace("HARPS_", "").replace(".txt", "")
        elif "UVES-POP" in spectraname:
            instrument = "UVES-POP"
            spectraname = spectraname.replace("UVES-POP_", "").replace(".txt", "")
        elif "UVES" in spectraname:
            instrument = "UVES"
            spectraname = spectraname.replace("UVES_", "").replace(".txt", "")
        else:
            instrument = "FOCES"
            spectraname = spectraname.replace(".txt", "")
        spectraname = spectraname.replace("_v2", "")
        spectraname = spectraname.replace("-1_Ceres", " Ceres")
        spectraname = spectraname.replace("-2_Ganymede", " Ganymede")
        spectraname = spectraname.replace("-3_Vesta", " Vesta")
        spectraname = spectraname.replace("BD-4_3208", "BD-4 3208")
        spectraname = spectraname.replace("BD+26_3578", "BD+26 3578")
        all_spectra_name.append(spectraname)
        if instrument not in instruments:
            instruments.append(instrument)
        idx_instrument = instruments.index(instrument) + 1
        total_string = f"{spectraname}$^{idx_instrument}$ & ${row[1]['teff'] * 1000:.0f} \\pm {row[1]['teff_std'] * 1000 + fit_systematic_error.teff_error:.0f}$ & ${row[1]['logg']:.2f} \\pm {row[1]['logg_std'] + fit_systematic_error.logg_error:.2f}$ & ${row[1]['vmic']:.2f} \\pm {row[1]['vmic_std']:.2f}$ & ${row[1]['feh']:.2f} \\pm {row[1]['feh_std'] + fit_systematic_error.feh_error:.2f}$  & ${row[1]['vsini']:.2f} \\pm {row[1]['vsini_std']:.2f}$ "
        #total_string = f"{spectraname}$^{idx_instrument}$ & ${row[1]['teff'] * 1000:.0f}$ & ${row[1]['logg']:.2f}$ & ${row[1]['vmic']:.2f}$ & ${row[1]['feh']:.2f}$  & ${row[1]['vsini']:.2f}$ "
        for el in elements1:
            if el not in row[1] or (el + '_std') not in row[1]:
                total_string += "& -"
            else:
                element_value = row[1][el]
                element_std = row[1][el + '_std']
                element_value_tsfitpy = row[1][f"{el}_tsfitpy"]
                if element_std < 0 or element_std > 0.5 or np.isnan(element_value_tsfitpy):
                    total_string += "& -"
                else:
                    total_string += f"& ${element_value:.2f} \\pm {element_std + fit_systematic_error.abundance_errors[el]:.2f}$"
                    #total_string += f"& ${element_value:.2f}$"
        total_string += " \\\\"
        print(total_string)
    print(len(set(all_spectra_name)))
    print(instruments)
    print("")

    for row in data.iterrows():
        spectraname = row[1]['spectraname']
        instrument = ""
        if "NARVAL" in spectraname:
            instrument = "NARVAL"
            spectraname = spectraname.replace("NARVAL_", "").replace(".txt", "")
        elif "HARPS" in spectraname:
            instrument = "HARPS"
            spectraname = spectraname.replace("HARPS_", "").replace(".txt", "")
        elif "UVES-POP" in spectraname:
            instrument = "UVES-POP"
            spectraname = spectraname.replace("UVES-POP_", "").replace(".txt", "")
        elif "UVES" in spectraname:
            instrument = "UVES"
            spectraname = spectraname.replace("UVES_", "").replace(".txt", "")
        else:
            instrument = "FOCES"
            spectraname = spectraname.replace(".txt", "")
        spectraname = spectraname.replace("_v2", "")
        spectraname = spectraname.replace("-1_Ceres", " Ceres")
        spectraname = spectraname.replace("-2_Ganymede", " Ganymede")
        spectraname = spectraname.replace("-3_Vesta", " Vesta")
        spectraname = spectraname.replace("BD-4_3208", "BD-4 3208")
        spectraname = spectraname.replace("BD+26_3578", "BD+26 3578")
        all_spectra_name.append(spectraname)
        if instrument not in instruments:
            instruments.append(instrument)
        idx_instrument = instruments.index(instrument) + 1
        total_string = f"{spectraname}$^{idx_instrument}$ "
        #total_string = f"{spectraname}$^{idx_instrument}$ & ${row[1]['teff'] * 1000:.0f}$ & ${row[1]['logg']:.2f}$ & ${row[1]['vmic']:.2f}$ & ${row[1]['feh']:.2f}$  & ${row[1]['vsini']:.2f}$ "
        for el in elements2:
            if el not in row[1] or (el + '_std') not in row[1]:
                total_string += "& -"
            else:
                element_value = row[1][el]
                element_std = row[1][el + '_std']
                element_value_tsfitpy = row[1][f"{el}_tsfitpy"]
                if element_std < 0 or element_std > 0.5 or np.isnan(element_value_tsfitpy):
                    total_string += "& -"
                else:
                    total_string += f"& ${element_value:.2f} \\pm {element_std + fit_systematic_error.abundance_errors[el]:.2f}$"
                    #total_string += f"& ${element_value:.2f}$"
        total_string += " \\\\"
        print(total_string)
    print("")

    for row in data.iterrows():
        spectraname = row[1]['spectraname']
        instrument = ""
        if "NARVAL" in spectraname:
            instrument = "NARVAL"
            spectraname = spectraname.replace("NARVAL_", "").replace(".txt", "")
        elif "HARPS" in spectraname:
            instrument = "HARPS"
            spectraname = spectraname.replace("HARPS_", "").replace(".txt", "")
        elif "UVES-POP" in spectraname:
            instrument = "UVES-POP"
            spectraname = spectraname.replace("UVES-POP_", "").replace(".txt", "")
        elif "UVES" in spectraname:
            instrument = "UVES"
            spectraname = spectraname.replace("UVES_", "").replace(".txt", "")
        else:
            instrument = "FOCES"
            spectraname = spectraname.replace(".txt", "")
        spectraname = spectraname.replace("_v2", "")
        spectraname = spectraname.replace("-1_Ceres", " Ceres")
        spectraname = spectraname.replace("-2_Ganymede", " Ganymede")
        spectraname = spectraname.replace("-3_Vesta", " Vesta")
        spectraname = spectraname.replace("BD-4_3208", "BD-4 3208")
        spectraname = spectraname.replace("BD+26_3578", "BD+26 3578")
        all_spectra_name.append(spectraname)
        if instrument not in instruments:
            instruments.append(instrument)
        idx_instrument = instruments.index(instrument) + 1
        total_string = f"{spectraname}$^{idx_instrument}$ "
        #total_string = f"{spectraname}$^{idx_instrument}$ & ${row[1]['teff'] * 1000:.0f}$ & ${row[1]['logg']:.2f}$ & ${row[1]['vmic']:.2f}$ & ${row[1]['feh']:.2f}$  & ${row[1]['vsini']:.2f}$ "
        for el in elements3:
            if el not in row[1] or (el + '_std') not in row[1]:
                total_string += "& -"
            else:
                element_value = row[1][el]
                element_std = row[1][el + '_std']
                element_value_tsfitpy = row[1][f"{el}_tsfitpy"]
                if element_std < 0 or element_std > 0.5 or np.isnan(element_value_tsfitpy):
                    total_string += "& -"
                else:
                    total_string += f"& ${element_value:.2f} \\pm {element_std + fit_systematic_error.abundance_errors[el]:.2f}$"
                    #total_string += f"& ${element_value:.2f}$"
        total_string += " \\\\"
        print(total_string)