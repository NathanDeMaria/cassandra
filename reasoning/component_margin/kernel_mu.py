"""The parts' margin as a kernel: key numbers from the cumulative logit, location and spread from mu and sd.

    python kernel_mu.py cm_NAME [against]

The release's kernel is a cumulative logit on every integer margin with a
win probability's logit as its one covariate. The parts hand out a mean and
an sd per game, so the kernel can take those instead:

    P(margin <= k) = sigmoid((c_k - b mu / 10) * exp(-g z)),   z = log(sd / median sd)

the free symmetric cutpoints keeping the key numbers, `b` the mean's slope,
`g` how far a game's own sd widens it (0 is the location-only kernel). Fit
on every game through 2021, as `compare.py` fits the published kernel, and
scored on FBS v FBS games, paired with the published model's kernel.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cassandra.prob_to_margin.kernel import MarginKernelFitter, _cutpoints, _sigmoid

sys.path.insert(0, str(Path(__file__).parent))
from compare import TRAIN_THROUGH, kernel_scores, load  # noqa: E402

K = 60


def scores_from_cdfs(cdfs: np.ndarray, margins: np.ndarray) -> dict[str, np.ndarray]:
    pmfs = np.diff(cdfs, axis=1, prepend=0.0)
    cells = np.clip(np.rint(margins).astype(int), -K, K) + K
    realized = np.arange(2 * K)[None, :] >= cells[:, None]
    return {
        "log_loss": -np.log(np.maximum(pmfs[np.arange(len(cells)), cells], 1e-300)),
        "rps": ((cdfs[:, :-1] - realized) ** 2).sum(axis=1),
    }


def cdfs_of(params: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    cut = _cutpoints(params[:K])
    b, g = params[K], params[K + 1]
    logits = (cut[None, :] - b * x[:, None]) * np.exp(-g * z)[:, None]
    return np.concatenate([_sigmoid(logits), np.ones((len(x), 1))], axis=1)


def fit_location_scale(x: np.ndarray, z: np.ndarray, margins: np.ndarray, start: np.ndarray) -> np.ndarray:
    cells = np.clip(np.rint(margins).astype(int), -K, K) + K
    rows = np.arange(len(cells))

    def nll(params: np.ndarray) -> float:
        pmfs = np.diff(cdfs_of(params, x, z), axis=1, prepend=0.0)
        return -np.log(np.maximum(pmfs[rows, cells], 1e-300)).mean()

    # The cutpoints and slope from the location-only fit, then b and g
    # together, then everything from there.
    two = minimize(lambda bg: nll(np.concatenate([start[:K], bg])), [start[K], 0.0], method="Nelder-Mead")
    full = np.concatenate([start[:K], two.x])
    refined = minimize(nll, full, method="L-BFGS-B", options={"maxiter": 60})
    return refined.x if refined.fun < two.fun else full


def main() -> None:
    name = sys.argv[1]
    against = sys.argv[2] if len(sys.argv) > 2 else "batch_nc_success_cap_p"
    a, b = load(name), load(against)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    a = a[a.mu.notna()]
    b = b.loc[a.index]
    train = (a.year <= TRAIN_THROUGH).to_numpy()
    m = a.team1_mov.to_numpy()
    x = a.mu.to_numpy() / 10
    z = np.log(a.sd.to_numpy() / np.median(a.sd.to_numpy()[train]))

    published = kernel_scores(b)
    own_prob = kernel_scores(a)
    location = MarginKernelFitter().fit(1 / (1 + np.exp(-x[train])), m[train])
    start = np.append(location._log_gaps, [location.slope])
    loc_cdfs = cdfs_of(np.append(start, 0.0), x, z)
    params = fit_location_scale(x[train], z[train], m[train], start)
    ls_cdfs = cdfs_of(params, x, z)
    variants = {
        "kernel on its own win prob": own_prob,
        "kernel on mu": scores_from_cdfs(loc_cdfs, m),
        f"kernel on mu and sd (b {params[K]:.3f}, g {params[K + 1]:+.3f})": scores_from_cdfs(ls_cdfs, m),
    }
    print(f"{name} vs {against}'s kernel, FBS v FBS; log loss 1e-3 nats, RPS 1e-2; negative favors {name}")
    print(f"  location-only slope {location.slope:.3f} per 10 points of mu")
    for label, scored in variants.items():
        cells = [f"{label:44s}"]
        for window, rows in (("2014-21", a.year.between(2014, TRAIN_THROUGH)), ("held out", a.year > TRAIN_THROUGH)):
            rows = (rows & a.fbs).to_numpy()
            for metric, unit in (("log_loss", 1e3), ("rps", 1e2)):
                diff = scored[metric][rows] - published[metric][rows]
                t = diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))
                cells.append(f"{window} {metric} {diff.mean() * unit:+6.2f} (t {t:+4.1f})")
        print("  " + " | ".join(cells))


if __name__ == "__main__":
    main()
