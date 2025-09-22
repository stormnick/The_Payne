from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 12.08.25

if __name__ == '__main__':
    path_linemasks = "/Users/storm/PycharmProjects/payne/lines_info/"
    # get all files in the path_linemasks that end with .csv
    linemask_files = [f for f in os.listdir(path_linemasks) if f.endswith('.csv')]

    all_linemasks = []
    for linemask_file in linemask_files:
        # read the csv file
        linemask_df = pd.read_csv(os.path.join(path_linemasks, linemask_file))
        linemask_df["element"] = linemask_file.split(".")[0]  # assuming the element is the first part of the filename
        all_linemasks.append(linemask_df)

    all_linemasks_df = pd.concat(all_linemasks, ignore_index=True)
    # drop column
    all_linemasks_df.drop(columns=["Unnamed: 0", "Unnamed: 3", "Bluewidth"], inplace=True, errors='ignore')  # in case the column exists
    # replace any empty values in ionisation column with 1
    all_linemasks_df['ionisation'] = all_linemasks_df['ionisation'].replace(np.nan, 1)
    # convert ionisation column to int
    all_linemasks_df['ionisation'] = all_linemasks_df['ionisation'].astype(int)
    # capitalize the first letter of each element in the element column
    all_linemasks_df['element'] = all_linemasks_df['element'].str.capitalize()
    # convert ionisation to I or II (e.g. 1 -> I, 2 -> II)
    all_linemasks_df['ionisation'] = all_linemasks_df['ionisation'].replace({1: 'I', 2: 'II', 3: 'III'})
    all_linemasks_df['element'] = all_linemasks_df['element'] + " " + all_linemasks_df['ionisation'].astype(str)
    # drop ionisation column
    all_linemasks_df.drop(columns=["ionisation"], inplace=True)
    # isotope to int
    all_linemasks_df['isotope'] = all_linemasks_df['isotope'].replace(np.nan, 0)
    all_linemasks_df['isotope'] = all_linemasks_df['isotope'].astype(int)
    print(all_linemasks_df)
    # save to csv
    all_linemasks_df.to_csv("all_linemasks.csv", index=False)