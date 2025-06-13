from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 13.06.25



if __name__ == '__main__':
    initial_learning_rate = [1e-4, 3e-4, 1e-3]
    final_loss = [2.2132e-05, 1.32160e-05, 1.40844e-05]

    plt.figure(figsize=(10, 6))
    plt.plot(initial_learning_rate, final_loss, marker='o', linestyle='-', color='k')
    plt.xscale('log')
    plt.xlabel('Initial Learning Rate')
    plt.ylabel('Final Loss')
    plt.title(f'Final Loss vs Initial Learning Rate')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()