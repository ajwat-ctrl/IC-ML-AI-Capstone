"""
bbo_core.py
Core building blocks for the IC ML & AI Black-Box Optimisation capstone.

Pipeline for each function, each round:
    1. load_function_data()   -> starter .npy data + your logged portal results
    2. fit a surrogate model  -> Gaussian Process (mean + uncertainty)
    3. generate candidates    -> global random points + local points near the best so far
    4. score with acquisition -> UCB or Expected Improvement
    5. check bounds + format  -> 'x1-x2-...-xn' string for the portal
"""

import json
import os
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Portal rule (Capstone FAQ): every value must start with 0 and have six decimals,
# i.e. 0.000000 <= x <= 0.999999.
LOWER, UPPER = 0.0, 0.999999

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "Data")
SUBMISSIONS_FILE = os.path.join(DATA_DIR, "submissions.json")


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def load_submissions(path=SUBMISSIONS_FILE):
    """Read the log of real portal results you have recorded so far."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_function_data(fn, data_dir=DATA_DIR, submissions=None):
    """
    Return (X, y, n_initial) for function `fn` (1-8).

    X, y combine the official starter data with every real portal result logged in
    submissions.json, in order. n_initial is the number of starter points, so plots
    can tell starter data apart from your own queries.
    """
    folder = os.path.join(data_dir, f"function_{fn}")
    X = np.load(os.path.join(folder, "initial_inputs.npy"))
    y = np.load(os.path.join(folder, "initial_outputs.npy")).astype(float).ravel()
    n_initial = len(y)

    if submissions is None:
        submissions = load_submissions()
    entries = sorted(submissions.get(f"function_{fn}", []), key=lambda e: e["round"])
    if entries:
        X_new = np.array([e["x"] for e in entries], dtype=float)
        y_new = np.array([e["y"] for e in entries], dtype=float)
        if X_new.shape[1] != X.shape[1]:
            raise ValueError(
                f"Function {fn}: logged inputs have {X_new.shape[1]} values, "
                f"expected {X.shape[1]}."
            )
        X = np.vstack([X, X_new])
        y = np.concatenate([y, y_new])
    return X, y, n_initial


# ---------------------------------------------------------------------------
# 2. Output transforms (help the GP cope with awkward output scales)
# ---------------------------------------------------------------------------
def transform_y(y, method):
    """
    'none'     : use outputs as they are
    'log1p'    : log(1 + y) for positive outputs spanning several orders of magnitude
    'signedlog': sign(y) * log(1 + |y|) for outputs mixing tiny and large values
    'rank'     : replace values by their rank, scaled to [0, 1]; robust to extreme spikes
    The transforms keep the ordering of y, so maximising the transformed value still
    maximises the true output.
    """
    y = np.asarray(y, dtype=float)
    if method == "none":
        return y
    if method == "log1p":
        return np.log1p(y - min(0.0, y.min()))
    if method == "signedlog":
        return np.sign(y) * np.log1p(np.abs(y))
    if method == "rank":
        ranks = y.argsort().argsort().astype(float)
        return ranks / max(len(y) - 1, 1)
    raise ValueError(f"Unknown transform: {method}")


# ---------------------------------------------------------------------------
# 3. Surrogate model
# ---------------------------------------------------------------------------
def fit_gp(X, y_t, nu=2.5, noise=True, seed=0):
    """
    Gaussian Process with an anisotropic Matern kernel (one length scale per input),
    so the model can learn that some inputs matter more than others.
    A WhiteKernel term lets it treat part of the variation as noise.
    """
    d = X.shape[1]
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.full(d, 0.3), length_scale_bounds=(1e-2, 1e1), nu=nu
    )
    if noise:
        kernel += WhiteKernel(1e-3, (1e-6, 1e-1))
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=8, random_state=seed
    )
    gp.fit(X, y_t)
    return gp


def loo_check(X, y_t, nu=2.5, noise=True):
    """
    Leave-one-out check: refit without each point and predict it.
    Returns (mean absolute error, calibration ratio = MAE / mean predicted std).
    A ratio near 1 means the GP's uncertainty is roughly honest; much above 1 means
    it is overconfident. With small datasets treat this as a rough signal only.
    """
    errors, stds = [], []
    for i in range(len(y_t)):
        mask = np.arange(len(y_t)) != i
        gp = fit_gp(X[mask], y_t[mask], nu=nu, noise=noise)
        mu, sd = gp.predict(X[i : i + 1], return_std=True)
        errors.append(abs(mu[0] - y_t[i]))
        stds.append(sd[0])
    mae = float(np.mean(errors))
    return mae, mae / max(float(np.mean(stds)), 1e-12)


# ---------------------------------------------------------------------------
# 4. Candidates and acquisition
# ---------------------------------------------------------------------------
def generate_candidates(X, y, n_global=20000, n_local=10000, local_scale=0.05, seed=0):
    """
    Two kinds of candidate points:
      - global: uniform over the whole valid box (exploration)
      - local : small Gaussian perturbations around the best observed points (exploitation)
    All candidates are clipped to the portal's valid range.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    global_pts = rng.uniform(LOWER, UPPER, size=(n_global, d))

    top = X[np.argsort(y)[-3:]]                      # three best points so far
    centres = top[rng.integers(0, len(top), n_local)]
    local_pts = centres + rng.normal(0, local_scale, size=(n_local, d))

    return np.clip(np.vstack([global_pts, local_pts]), LOWER, UPPER)


def ucb(mu, sd, kappa):
    """Upper Confidence Bound: predicted value plus an exploration bonus."""
    return mu + kappa * sd


def expected_improvement(mu, sd, best, xi=0.01):
    """Expected amount by which a point beats the current best (minus margin xi)."""
    sd = np.maximum(sd, 1e-12)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def drop_near_duplicates(candidates, X, min_dist=1e-3):
    """Remove candidates that almost repeat an already-observed input."""
    dists = np.min(np.linalg.norm(candidates[:, None, :] - X[None, :, :], axis=2), axis=1)
    return candidates[dists > min_dist]


# ---------------------------------------------------------------------------
# 5. Submission helpers
# ---------------------------------------------------------------------------
def check_bounds(x):
    """Raise an error if any value would be rejected by the portal."""
    x = np.asarray(x, dtype=float)
    if np.any(x < LOWER) or np.any(x > UPPER):
        raise ValueError(f"Query out of bounds [{LOWER}, {UPPER}]: {x}")
    return x


def format_query(x):
    """Portal format: six decimals, hyphen-separated, no spaces."""
    return "-".join(f"{v:.6f}" for v in check_bounds(x))


# ---------------------------------------------------------------------------
# Full step for one function
# ---------------------------------------------------------------------------
def propose_query(X, y, config, seed=0, run_loo=False):
    """
    Fit the surrogate, score candidates and return (query, diagnostics dict).
    `config` keys: transform, nu, noise, acquisition ('ucb' or 'ei'), kappa, xi, local_scale.
    """
    y_t = transform_y(y, config["transform"])
    gp = fit_gp(X, y_t, nu=config["nu"], noise=config["noise"], seed=seed)

    cands = generate_candidates(X, y, local_scale=config["local_scale"], seed=seed)
    cands = drop_near_duplicates(cands, X)

    # Predict in batches to keep memory use low
    mu = np.empty(len(cands))
    sd = np.empty(len(cands))
    for start in range(0, len(cands), 5000):
        m, s = gp.predict(cands[start : start + 5000], return_std=True)
        mu[start : start + 5000], sd[start : start + 5000] = m, s

    if config["acquisition"] == "ucb":
        scores = ucb(mu, sd, config["kappa"])
    else:
        scores = expected_improvement(mu, sd, best=y_t.max(), xi=config["xi"])

    i = int(np.argmax(scores))
    query = check_bounds(cands[i])

    diagnostics = {
        "n_points": len(y),
        "best_observed": float(y.max()),
        "best_observed_x": X[int(np.argmax(y))].round(6).tolist(),
        "pred_mean_transformed": float(mu[i]),
        "pred_std_transformed": float(sd[i]),
        "kernel": str(gp.kernel_),
    }
    if run_loo:
        mae, ratio = loo_check(X, y_t, nu=config["nu"], noise=config["noise"])
        diagnostics["loo_mae"] = mae
        diagnostics["loo_calibration_ratio"] = ratio
    return query, diagnostics
