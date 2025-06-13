from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from convolve_vmac import *

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 08.05.25

wave_rsg1, flux_rsg1 = np.loadtxt("/Users/storm/Downloads/2025-05-08-10-51-04_0.9940585176299548_LTE_rsg_test1_8may2025.txt/0.spec", dtype=float, usecols=(0,2), unpack=True)
wave_rsg2, flux_rsg2 = np.loadtxt("/Users/storm/Downloads/2025-05-08-10-51-04_0.9940585176299548_LTE_rsg_test1_8may2025.txt/1.spec", dtype=float, usecols=(0,2), unpack=True)

wave_rsg1, flux_rsg1 = conv_res(wave_rsg1, flux_rsg1, 20000)
wave_rsg1, flux_rsg1 = conv_macroturbulence(wave_rsg1, flux_rsg1, 15)
wave_rsg2, flux_rsg2 = conv_res(wave_rsg2, flux_rsg2, 20000)
wave_rsg2, flux_rsg2 = conv_macroturbulence(wave_rsg2, flux_rsg2, 15)

plt.figure(figsize=(10,4))
plt.plot(wave_rsg1, flux_rsg1 / 3.14, label="RSG1", linewidth=0.05, c='k')
plt.xlim(np.min(wave_rsg1), np.max(wave_rsg1))
plt.ylim(0, 2e5)
plt.xlabel("Wavelength (AA)")
plt.ylabel("Flux [cgs]")
plt.show()
plt.figure(figsize=(10,4))
plt.plot(wave_rsg2, flux_rsg2 / 3.14, label="RSG2", linewidth=0.05, c='k')
plt.xlim(np.min(wave_rsg2), np.max(wave_rsg2))
plt.ylim(0, 8e5)
plt.xlabel("Wavelength (AA)")
plt.ylabel("Flux [cgs]")
plt.show()