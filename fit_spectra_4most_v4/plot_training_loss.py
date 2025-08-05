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

    final_loss_silu_100k = [5.92878e-06]
    final_loss_silu_batch0 = [5.18703e-06]

    xy_text = [(10, 7), (10, 10), (-7, 18), (-12, 7)]
    xy_text2 = [(10, 7), (10, 10), (-7, -11), (-12, 7)]

    plt.figure(figsize=(7, 5))
    fontsize = 14
    plt.plot(initial_learning_rate_relu, final_loss_relu, marker='o', linestyle='-', color='k', label='ReLU')
    plt.plot(initial_learning_rate_silu, final_loss_silu, marker='o', linestyle='-', color='r', label='SiLU')
    # write loss next to each point
    for i, txt in enumerate(final_loss_relu):
        plt.annotate(f'{txt:.2e}', (initial_learning_rate_relu[i], final_loss_relu[i]), textcoords="offset points", xytext=xy_text[i], ha='center', fontsize=fontsize-2)
    for i, txt in enumerate(final_loss_silu):
        plt.annotate(f'{txt:.2e}', (initial_learning_rate_silu[i], final_loss_silu[i]), textcoords="offset points", xytext=xy_text2[i], ha='center', fontsize=fontsize-2)

    plt.scatter([1e-3], final_loss_silu_100k, marker='x', color='r', label='SiLU Half Epochs', s=40)
    plt.annotate(f'{final_loss_silu_100k[0]:.2e}', (1e-3, final_loss_silu_100k[0]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=fontsize-4)
    plt.scatter([1e-3], final_loss_silu_batch0, marker='v', color='r', label='SiLU Half Training Spectra', s=40)
    plt.annotate(f'{final_loss_silu_batch0[0]:.2e}', (1e-3, final_loss_silu_batch0[0]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=fontsize-4)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Initial Learning Rate', fontsize=fontsize)
    plt.ylabel('Final Loss', fontsize=fontsize)
    plt.title(f'Final Loss vs Initial Learning Rate', fontsize=fontsize+4)
    #plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # xticks = [1e-4, 3e-4, 1e-3, 3e-3]
    #plt.xticks(initial_learning_rate_relu, [f'{lr:.0e}' for lr in initial_learning_rate_relu], fontsize=fontsize)
    #plt.xticks(initial_learning_rate_relu, [f'{lr:.1d}' for lr in initial_learning_rate_relu], fontsize=fontsize)
    plt.xticks(initial_learning_rate_relu, [f'{lr}' for lr in initial_learning_rate_relu], fontsize=fontsize)
    # yticks fontsize
    plt.tick_params(axis='both', which='both', labelsize=fontsize)
    plt.ylim(3.5e-6, 2.9e-5)
    yticks_to_do = [4e-6, 6e-6, 1e-5, 2e-5]
    plt.yticks(yticks_to_do, [f'{ytick:.0e}' for ytick in yticks_to_do], fontsize=fontsize)
    #plt.yticks(yticks_to_do, fontsize=fontsize)
    # .tick_params(axis='both', which='major', labelsize=11)
    plt.legend(fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.0, 0.0), frameon=False)
    plt.tight_layout()
    plt.savefig("../plots/final_loss_vs_initial_learning_rate.pdf", bbox_inches='tight')
    plt.show()