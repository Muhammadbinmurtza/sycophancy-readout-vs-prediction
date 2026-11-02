"""Question-level bootstrap 95% CIs (2,000 resamples, seed 0) for the pilot's headline numbers.

    E1   simulated in-distribution: G-probe, question-level 5-fold CV, resid L16 and top-16 heads
    E6   transfer: TruthfulQA-trained G-probe scored on real episodes (CI vs CC), same read-outs
    UB   in-domain upper bound on real episodes, resid L15 (CI vs CC) and pre-reply L4

Layers L16 / L15 / L4 were chosen as the best layer in earlier runs, so their point estimates
are optimistically selected; the CIs quantify sampling noise only, not selection.

    python ci_bootstrap.py cache_qwen1.5b.pt results_qwen1.5b.json results_e6_qwen1.5b_feats.npz ci_qwen1.5b.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from analyze_confound import K, SEED, build_sets, probe

B = 2000


def boot_auc(y, s, groups, rng):
    """AUROC with a question-level (cluster) bootstrap CI."""
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    stats = []
    for _ in range(B):
        take = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        if len(set(y[take])) == 2:
            stats.append(roc_auc_score(y[take], s[take]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"auc": float(roc_auc_score(y, s)), "ci95": [float(lo), float(hi)]}


def cv_scores(X, y, groups, folds):
    s = np.zeros(len(y))
    for tq in folds:
        te = np.isin(groups, tq)
        s[te] = probe().fit(X[~te], y[~te]).decision_function(X[te])
    return s


def main(cache_path, results_path, feats_path, out_path):
    rng = np.random.default_rng(SEED)
    blob = torch.load(cache_path)
    qs, G, _, _ = build_sets(blob["kinds"], blob["qidx"].tolist())
    gi, gy, gq = G
    folds = np.array_split(np.random.default_rng(SEED).permutation(qs), K)
    resid = blob["resid_mean"].float().numpy()
    heads = blob["heads_last"].float().numpy()
    dh = blob["head_dim"]
    top = json.load(open(results_path))["heads_summary"]["top16_by_genadi_acc"]

    d = np.load(feats_path)
    kinds, qids = d["kinds"], d["qids"]
    m = np.isin(kinds, ["real_CI", "real_CC"])
    y_real, q_real = (kinds[m] == "real_CI").astype(int), qids[m]

    out = {}
    # E1 and E6 at resid L16
    X = resid[:, 16]
    out["E1_sim_resid_L16"] = boot_auc(gy, cv_scores(X[gi], gy, gq, folds), gq, rng)
    s_real = probe().fit(X[gi], gy).decision_function(d["ans_resid"][m, 16])
    out["E6_real_resid_L16_answer"] = boot_auc(y_real, s_real, q_real, rng)

    # top-16 heads: average the z-scored per-head scores into one ensemble score
    sim_scores, real_scores = [], []
    for key in top:
        l, h = map(int, key.split("_"))
        sl = slice(h * dh, (h + 1) * dh)
        sc = cv_scores(heads[gi][:, l, sl], gy, gq, folds)
        sim_scores.append((sc - sc.mean()) / sc.std())
        rs = probe().fit(heads[gi][:, l, sl], gy).decision_function(d["ans_heads"][m][:, l, sl])
        real_scores.append((rs - rs.mean()) / rs.std())
    out["E1_sim_top16_ensemble"] = boot_auc(gy, np.mean(sim_scores, 0), gq, rng)
    out["E6_real_top16_ensemble_answer"] = boot_auc(y_real, np.mean(real_scores, 0), q_real, rng)

    # in-domain upper bound on real episodes
    uq = np.unique(q_real)
    rfolds = np.array_split(np.random.default_rng(SEED).permutation(uq), K)
    out["UB_real_indomain_resid_L15_answer"] = boot_auc(
        y_real, cv_scores(d["ans_resid"][m, 15], y_real, q_real, rfolds), q_real, rng)
    out["UB_real_indomain_resid_L4_prereply"] = boot_auc(
        y_real, cv_scores(d["pre_resid"][m, 4], y_real, q_real, rfolds), q_real, rng)

    for k, v in out.items():
        print(f"{k:38s} AUROC {v['auc']:.3f}  95% CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]", flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
