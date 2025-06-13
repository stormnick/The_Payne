from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("MacOSX")
# plt.style.use("/Users/storm/PycharmProjects/bensby_3d_nlte/Bergemann2020.mplstyle")

# Created by storm at 11.06.25

if __name__ == '__main__':
    labels_path = "/Users/storm/Downloads/labels_batch01.csv"

    # ---------- 1. Read catalogue ----------
    df = pd.read_csv(labels_path)

    df["teff"] = df["teff"] * 1000

    required = {"teff", "logg", "feh"}
    if not required.issubset(df.columns.str.lower()):
        raise ValueError(f"CSV must contain the columns: {', '.join(required)}")

    # ---------- 2. Define bin edges ----------
    teff_edges = np.arange(3500, 8000 + 500, 500)  # 3500, 4000, …, 8000
    logg_edges = np.arange(0.5, 5.0 + 0.5, 0.5)  # 0.5, 1.0, …, 5.0

    teff_labels = [f"{lo}-{hi}" for lo, hi in zip(teff_edges[:-1], teff_edges[1:])]
    logg_labels = [f"{lo}-{hi}" for lo, hi in zip(logg_edges[:-1], logg_edges[1:])]

    # keep only rows inside the bounding box
    sel = (
            (df["teff"].between(teff_edges[0], teff_edges[-1], inclusive="left"))
            & (df["logg"].between(logg_edges[0], logg_edges[-1], inclusive="left"))
    )
    df = df.loc[sel].copy()

    # bin assignment (right edge *excluded* so that 4000 belongs to 4000-4500, etc.)
    df["teff_bin"] = pd.cut(df["teff"], bins=teff_edges, right=False, labels=teff_labels)
    df["logg_bin"] = pd.cut(df["logg"], bins=logg_edges, right=False, labels=logg_labels)

    # ---------- 3. Count stars in every box ----------
    counts = (
        df.pivot_table(index="logg_bin", columns="teff_bin", values="feh", aggfunc="count")
        .reindex(index=logg_labels, columns=teff_labels)  # keep empty boxes
        .fillna(0)
        .astype(int)
    )

    print("\n# Stars per (Teff, log g) box")
    print(counts.to_string())

    # ---------- 4. Plot Fe/H histograms ----------
    n_teff, n_logg = len(teff_labels), len(logg_labels)
    fig, axes = plt.subplots(
        n_teff, n_logg, figsize=(2.8 * n_logg, 2.3 * n_teff), sharex=True, sharey=True
    )

    for i, t_lbl in enumerate(teff_labels):
        for j, g_lbl in enumerate(logg_labels):
            ax = axes[i, j]

            subset = df[(df["teff_bin"] == t_lbl) & (df["logg_bin"] == g_lbl)]
            if subset.empty:
                ax.set_axis_off()  # nothing inside – hide subplot
                continue

            ax.hist(subset["feh"].dropna(), bins="auto")
            ax.set_title(f"{t_lbl} K\nlog g {g_lbl}\nN = {len(subset)}", fontsize=7)
            if i == n_teff - 1:
                ax.set_xlabel("[Fe/H]")
            if j == 0:
                ax.set_ylabel("Stars")

    fig.suptitle("[Fe/H] distribution in Teff–log g boxes", y=0.995, fontsize=14)
    fig.tight_layout()

    #if args.savefig:
    #    fig.savefig(args.savefig, dpi=300)
    #    print(f"\nFigure saved to {args.savefig}")

    plt.show()