from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 13.05.25

# ------------------------------------------------------------------
# 1.  helper ops
# ------------------------------------------------------------------
def relu(x):
    return np.maximum(x, 0.0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_jacobian_once(labels, W0, b0, W1, b1, W2, b2, W3, b3):
    """
    Return J (n_pixels × n_labels) for a single label vector.
    All arguments are 1-D or 2-D NumPy arrays.
    """
    # ---------- forward pass ----------
    z0 = W0 @ labels + b0               # (H1,)
    a0 = relu(z0)

    z1 = W1 @ a0 + b1                   # (H2,)
    a1 = relu(z1)

    z2 = W2 @ a1 + b2                   # (n_pixels,)
    a3 = relu(z2)

    z3 = W3 @ a3 + b3                   # (n_pixels,)
    y  = sigmoid(z3)

    # ---------- diagonal derivative masks ----------
    Dsigma = y * (1.0 - y)              # (n_pixels,)
    D2mask = (z2 > 0).astype(float)     # (n_pixels,)
    D1mask = (z1 > 0).astype(float)     # (H2,)
    D0mask = (z0 > 0).astype(float)     # (H1,)

    # ---------- Jacobian ----------
    # note: multiplying by a diag-matrix is element-wise "*"
    # shapes:         (n_pix,H2)  (H2,)   (H2,H1)  (H1,)   (H1,n_lab)
    J = (Dsigma[:,None] *  W3) @ (D2mask[:,None] *  W2) @ (D1mask[:,None] * W1) @ (D0mask[:,None] * W0)

    # final J: (n_pixels × n_labels)
    return J, y

def load_payne(path_model):
    """
    Loads the Payne model coefficients from a .npz file.
    :param path_model: Path to the .npz file containing the Payne model coefficients.
    :return: Returns a tuple containing the coefficients and the wavelength array.
    """
    tmp = np.load(path_model)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    w_array_3 = tmp["w_array_3"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    b_array_3 = tmp["b_array_3"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    # wavelength is the wavelength array corresponding to the Payne model in AA
    wavelength = tmp["wavelength"]
    # labels are the label names corresponding to the Payne model, e.g. "teff", "logg", "feh", etc.
    labels = list(tmp["label_names"])
    tmp.close()
    # w_array are the weights, b_array are the biases
    # x_min and x_max are the minimum and maximum values for scaling the labels
    payne_coeffs = (w_array_0, w_array_1, w_array_2, w_array_3,
                    b_array_0, b_array_1, b_array_2, b_array_3,
                    x_min, x_max)
    return payne_coeffs, wavelength, labels


if __name__ == '__main__':
    path_model = "/Users/storm/PycharmProjects/payne/test_network/payne_ts_4most_hr_may2025_batch01_medium_test2training_reducedlogg_altarch_2025-06-16-06-28-26.npz"
    payne_coeffs, wavelength, labels = load_payne(path_model)
    w_array_0, w_array_1, w_array_2, w_array_3, b_array_0, b_array_1, b_array_2, b_array_3, x_min, x_max = payne_coeffs

    # ------------------------------------------------------------------
    # 2.  load nets & labels ------------------------------------------------
    W0, W1, W2, W3   = w_array_0, w_array_1, w_array_2, w_array_3
    b0, b1, b2, b3   = b_array_0, b_array_1, b_array_2, b_array_3
    label_names = labels
    labels = -np.zeros(len(labels)) / 2  # dummy labels
    labels[2] = -0.2
    labels[1] = -0.45
    λ            = wavelength                            # shape (n_pixels,)

    J, y_pred = compute_jacobian_once(labels, W0, b0, W1, b1, W2, b2, W3, b3)
    absJ      = np.abs(J)                                # importance score

    # ------------------------------------------------------------------
    # 3a.  heat-map: |J|  ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.imshow(absJ.T, aspect='auto',
               extent=[λ.min(), λ.max(), 0, len(labels)-1])
    plt.xlabel("Wavelength")
    plt.ylabel("Label index")
    plt.title("Absolute label→pixel sensitivity |∂y/∂x|")
    plt.colorbar(label="|Jacobian entry|")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 3b.  influence curves for first few labels ----------------------------
    for i in range(len(labels)):
        plt.figure(figsize=(10, 4))
        plt.plot(λ, absJ[:, i], label=f"label {label_names[i]}")
        plt.title("Pixel-wise influence of selected labels")
        plt.xlabel("Wavelength")
        plt.ylabel("|∂y/∂x|")
        plt.legend()
        plt.tight_layout()
        plt.show()
