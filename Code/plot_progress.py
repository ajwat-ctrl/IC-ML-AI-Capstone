"""
plot_progress.py
Plot every observed output and the best-so-far line for all eight functions.
The vertical dashed line separates the official starter data from your own queries.

Usage (from the repo root):
    python Code/plot_progress.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bbo_core import REPO_ROOT, load_function_data, load_submissions


def main():
    submissions = load_submissions()
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for fn, ax in zip(range(1, 9), axes.ravel()):
        X, y, n_init = load_function_data(fn, submissions=submissions)
        idx = np.arange(1, len(y) + 1)
        ax.scatter(idx[:n_init], y[:n_init], s=18, color="grey", label="starter data")
        if len(y) > n_init:
            ax.scatter(idx[n_init:], y[n_init:], s=30, color="tab:red", label="my queries")
        ax.plot(idx, np.maximum.accumulate(y), color="tab:blue", label="best so far")
        ax.axvline(n_init + 0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"Function {fn} ({X.shape[1]}D)")
        ax.set_xlabel("observation number")
        ax.set_ylabel("output")
        if fn == 1:
            ax.legend(fontsize=8)

    fig.suptitle("BBO capstone: observed outputs and best-so-far per function")
    fig.tight_layout()
    out = os.path.join(REPO_ROOT, "Results", "progress.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
