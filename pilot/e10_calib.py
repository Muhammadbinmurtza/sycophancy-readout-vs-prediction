"""E10 calibration and practical size of the prediction (secondary, added after the primary analysis).

Re-fits the primary E10 probe at its reported best layer only (same folds, residualisation and ridge) and
reports how well the out-of-fold predictions are calibrated and what they mean in P(cave) units:
  slope       OLS slope of actual on predicted residual P(cave) (1 = calibrated scale; <1 = over-confident)
  quintiles   mean actual residual P(cave) in each quintile of the prediction
  top-bottom  actual residual P(cave) of the top minus the bottom prediction quintile (percentage points)
  within-q    mean Spearman rho between predicted and actual over questions with >= 5 cells, and the share of
              those questions where the pushback predicted worst has an above-median actual P(cave)
  reliable    r^2 / interaction reliability: share of the reliable interaction variance explained

    python e10_calib.py qwen3_4b ../gpu_out/e10 ../gpu_out/qwen3_4b/e9_qwen3_4b ../gpu_out/e10/e10_qwen3_4b_p200.json out.json
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

from e10_analyse import K, SEED, load, pca, ridge_oof


def main(tag, e10dir, epdir, primary_json, out_path):
    prim = json.load(open(primary_json))
    b = prim["primary"]["best_layer"]
    cells, P, Pa, Pb, X, q, t = load(tag, e10dir, epdir)
    folds = np.array_split(np.random.default_rng(SEED).permutation(np.unique(q)), K)
    pred, targ = ridge_oof(pca(X[:, b]), P, q, t, folds, "fe")
    r = float(np.corrcoef(pred, targ)[0, 1])
    assert abs(r - prim["primary"]["r"]) < 1e-6, (r, prim["primary"]["r"])
    slope = float(np.polyfit(pred, targ, 1)[0])
    edges = np.quantile(pred, [0.2, 0.4, 0.6, 0.8])
    qi = np.digitize(pred, edges)
    quint = [float(targ[qi == k].mean()) for k in range(5)]
    rhos, worst_hit = [], []
    for g in np.unique(q):
        m = q == g
        if m.sum() >= 5 and np.std(targ[m]) > 0:
            rhos.append(spearmanr(pred[m], targ[m])[0])
            worst_hit.append(targ[m][np.argmax(pred[m])] > np.median(targ[m]))
    rel = prim["interaction_reliability"]["spearman_brown"]
    out = {"tag": tag, "layer": b, "r": r, "calibration_slope": slope,
           "quintile_mean_actual": quint, "top_minus_bottom_pp": 100 * (quint[-1] - quint[0]),
           "sd_actual_pp": 100 * float(targ.std()),
           "within_q_mean_spearman": float(np.mean(rhos)), "n_q_within": len(rhos),
           "within_q_worst_above_median": float(np.mean(worst_hit)),
           "reliable_variance_explained": r ** 2 / rel}
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:6])
