from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 13.06.25



if __name__ == '__main__':
    initial_learning_rate_relu = [1e-4, 3e-4, 1e-3, 3e-3]
    final_loss_relu = [2.2132e-05, 1.32160e-05, 9.92377e-06, 1.86755e-05]
    #initial_learning_rate_silu = [1e-4, 3e-4, 1e-3, 3e-3]
    #final_loss_silu = [2.2132e-05, 7.17515e-06, 9.24473e-06, 4.53383e-02]
    initial_learning_rate_silu = [1e-4, 3e-4, 1e-3, 3e-3]
    final_loss_silu = [1.2293e-05, 7.17515e-06, 4.13244e-06, 9.82680e-06]

    plt.figure(figsize=(7, 5))
    fontsize = 13
    plt.plot(initial_learning_rate_relu, final_loss_relu, marker='o', linestyle='-', color='k', label='ReLU Activation')
    plt.plot(initial_learning_rate_silu, final_loss_silu, marker='o', linestyle='-', color='r', label='SiLU Activation')
    # write loss next to each point
    for i, txt in enumerate(final_loss_relu):
        plt.annotate(f'{txt:.2e}', (initial_learning_rate_relu[i], final_loss_relu[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=fontsize-4)
    for i, txt in enumerate(final_loss_silu):
        plt.annotate(f'{txt:.2e}', (initial_learning_rate_silu[i], final_loss_silu[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=fontsize-4)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Initial Learning Rate', fontsize=fontsize)
    plt.ylabel('Final Loss', fontsize=fontsize)
    plt.title(f'Final Loss vs Initial Learning Rate', fontsize=fontsize)
    #plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # xticks = [1e-4, 3e-4, 1e-3, 3e-3]
    plt.xticks(initial_learning_rate_relu, [f'{lr:.0e}' for lr in initial_learning_rate_relu], fontsize=fontsize)
    # yticks fontsize
    plt.ylim(3.8e-6, 2.9e-5)
    plt.yticks(fontsize=fontsize)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../plots/final_loss_vs_initial_learning_rate.pdf", bbox_inches='tight')
    plt.show()