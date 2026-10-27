"""Gaussian-process Bayesian optimisation helpers for the BBO capstone.

Assumptions:
  * every input lies in [0, 1) and every function is MAXIMISED;
  * a query is submitted as hyphen-separated values with six decimals,
    e.g. "0.123456-0.654321" for a 2D function.

Per-function settings (see config.json) mirror the keys logged in
data/rounds/round_XX_queries.json: output transform, Matern nu, noise term,
acquisition function (ucb or ei), kappa / xi and the local search scale.
The transform definitions below were inferred from those logs and are a
reconstruction, not a copy of the original submission code.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

COLUMNS = ["function", "round", "query", "y", "rationale"]
UPPER = 0.999999


# ---------------------------------------------------------------- parsing / io
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


# ------------------------------------------------------------------ transforms
def transform_y(y: np.ndarray, kind: str = "none") -> np.ndarray:
    """Monotonic transforms of the outputs, applied before fitting the GP."""
    y = np.asarray(y, dtype=float)
    if kind == "none":
        return y
    if kind == "rank":       # ranks scaled to [0, 1]; for flat/spiky outputs
        return (rankdata(y) - 1) / max(len(y) - 1, 1)
    if kind == "signedlog":  # sign(y) * log(1 + |y|); for outputs of both signs and wide range
        return np.sign(y) * np.log1p(np.abs(y))
    if kind == "log1p":      # log(1 + y); for outputs spanning orders of magnitude
        return np.log1p(np.maximum(y, -0.999999))
    raise ValueError(f"unknown transform {kind!r}")


# ----------------------------------------------------------------------- model
def fit_gp(X: np.ndarray, y: np.ndarray, nu: float = 2.5, noise: bool = True,
           seed: int = 0) -> GaussianProcessRegressor:
    """GP with an ARD Matern kernel, optionally plus a learned noise term."""
    d = X.shape[1]
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(d), length_scale_bounds=(1e-2, 10.0), nu=nu)
    if noise:
        kernel = kernel + WhiteKernel(1e-3, (1e-6, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-8,
                                  n_restarts_optimizer=10, random_state=seed)
    gp.fit(X, y)
    return gp


def _lengthscales(gp):
    k = gp.kernel_
    for getter in (lambda k: k.k1.k2.length_scale, lambda k: k.k2.length_scale):
        try:
            return np.round(getter(k), 3).tolist()
        except AttributeError:
            continue
    return None


def _score(mu, sd, best, strategy, kappa, xi):
    if strategy == "ucb":
        return mu + kappa * sd
    if strategy == "ei":
        sd = np.maximum(sd, 1e-12)
        z = (mu - best - xi) / sd
        return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)
    raise ValueError(f"unknown strategy {strategy!r} (use 'ucb' or 'ei')")


def suggest(X: np.ndarray, y: np.ndarray, strategy: str = "ucb", kappa: float = 2.0,
            xi: float = 0.0, transform: str = "none", nu: float = 2.5, noise: bool = True,
            n_candidates: int = 20000, local_frac: float = 0.5,
            local_scale: float = 0.05, min_dist: float = 1e-3, seed: int = 0):
    """Propose the next query. Returns (x, info). Scores are in transformed units."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    yt = transform_y(y, transform)
    gp = fit_gp(X, yt, nu=nu, noise=noise, seed=seed)

    n_local = int(n_candidates * local_frac)
    best_x = X[np.argmax(y)]
    local = np.clip(best_x + rng.normal(0, local_scale, (n_local, d)), 0, UPPER)
    global_ = rng.random((n_candidates - n_local, d)) * UPPER
    cand = np.vstack([local, global_])

    # avoid re-querying points that are (almost) already observed
    dist = np.min(np.linalg.norm(cand[:, None, :] - X[None, :, :], axis=2), axis=1)
    cand = cand[dist >= min_dist]

    mu, sd = gp.predict(cand, return_std=True)
    score = _score(mu, sd, yt.max(), strategy, kappa, xi)
    i = int(np.argmax(score))

    info = {"pred_mean_transformed": float(mu[i]), "pred_std_transformed": float(sd[i]),
            "score": float(score[i]), "best_so_far": float(np.max(y)), "n_obs": int(len(y)),
            "lengthscales": _lengthscales(gp)}
    return cand[i], info
