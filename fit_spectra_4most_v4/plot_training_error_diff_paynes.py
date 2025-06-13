from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 10.06.25


if __name__ == '__main__':
    neurons = [900, 900, 2000, 300]
    final_error = [2.6230e-05, 1.77968e-05, 1.48436e-05, 3.0318e-05]
    training_size = [146_000, 146_000 * 2, 146_000 * 2, 146_000 * 2]
    names = ["Medium Payne", "Medium Payne", "Big Payne", "Small Payne"]

    plt.figure(figsize=(10, 6))
    training_size_plot = training_size[0:2]
    final_error_plot = final_error[0:2]
    plt.plot(training_size_plot, final_error_plot, marker='o', linestyle='-', color='blue', label=names[0])
    training_size_plot = training_size[2:3]
    final_error_plot = final_error[2:3]
    plt.plot(training_size_plot, final_error_plot, marker='o', linestyle='-', color='orange', label=names[2])
    training_size_plot = training_size[3:4]
    final_error_plot = final_error[3:4]
    plt.plot(training_size_plot, final_error_plot, marker='o', linestyle='-', color='green', label=names[3])
    plt.xscale('log')
    plt.xlabel('Training Size (k)')
    plt.ylabel('Final Error')
    plt.title(f'Final Error vs Training Size for {names[0]}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()