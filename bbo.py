"""Minimal Gaussian-process Bayesian optimisation helpers for the BBO capstone.

Assumptions (check against the capstone portal guide):
  * every input lies in [0, 1) and every function is MAXIMISED;
  * a query is submitted as hyphen-separated values with six decimals,
    e.g. "0.123456-0.654321" for a 2D function.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

COLUMNS = ["function", "round", "query", "y", "rationale"]
UPPER = 0.999999


def parse_query(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.strip().split("-")], dtype=float)


def format_query(x: np.ndarray) -> str:
    x = np.clip(np.asarray(x, dtype=float), 0.0, UPPER)
    return "-".join(f"{v:.6f}" for v in x)


def load_observations(path: str | Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return {function_id: (X, y)} from the long-format observations CSV."""
    rows = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if not r["function"] or r["y"] in ("", None):
                continue
            rows[int(r["function"])].append((parse_query(r["query"]), float(r["y"])))
    out = {}
    for f, items in rows.items():
        out[f] = (np.vstack([i[0] for i in items]), np.array([i[1] for i in items]))
    return out


def append_observation(path: str | Path, function: int, rnd: int, query: str,
                       y: float, rationale: str = "") -> None:
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLUMNS)
        w.writerow([function, rnd, query, y, rationale])


def fit_gp(X: np.ndarray, y: np.ndarray, seed: int = 0) -> GaussianProcessRegressor:
    """GP with an ARD Matern-5/2 kernel plus a learned noise term."""
    d = X.shape[1]
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-3, (1e-6, 1e-1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=10, random_state=seed)
    gp.fit(X, y)
    return gp


def _score(mu, sd, best, strategy, kappa):
    if strategy == "ucb":
        return mu + kappa * sd
    if strategy == "ei":
        sd = np.maximum(sd, 1e-12)
        z = (mu - best) / sd
        return (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
    raise ValueError(f"unknown strategy {strategy!r} (use 'ucb' or 'ei')")


def suggest(X: np.ndarray, y: np.ndarray, strategy: str = "ucb", kappa: float = 2.0,
            n_candidates: int = 20000, local_frac: float = 0.5,
            local_scale: float = 0.08, min_dist: float = 1e-3, seed: int = 0):
    """Propose the next query. Returns (x, info)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    gp = fit_gp(X, y, seed)

    n_local = int(n_candidates * local_frac)
    best_x = X[np.argmax(y)]
    local = np.clip(best_x + rng.normal(0, local_scale, (n_local, d)), 0, UPPER)
    global_ = rng.random((n_candidates - n_local, d)) * UPPER
    cand = np.vstack([local, global_])

    # avoid re-querying points that are (almost) already observed
    dist = np.min(np.linalg.norm(cand[:, None, :] - X[None, :, :], axis=2), axis=1)
    cand = cand[dist >= min_dist]

    mu, sd = gp.predict(cand, return_std=True)
    score = _score(mu, sd, y.max(), strategy, kappa)
    i = int(np.argmax(score))

    k = gp.kernel_
    try:
        lengthscales = k.k1.k2.length_scale
    except AttributeError:
        lengthscales = None
    info = {"pred_mean": float(mu[i]), "pred_std": float(sd[i]), "score": float(score[i]),
            "best_so_far": float(y.max()), "n_obs": int(len(y)),
            "lengthscales": None if lengthscales is None else np.round(lengthscales, 3).tolist()}
    return cand[i], info
