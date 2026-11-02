"""E10 follow-up: does the Genadi-style simulated-dialogue probe predict the E10 interaction?

Same cells and two-way FE residual P(cave) as e10_analyse.py. Score = G-probe trained on simulated dialogues
(question-disjoint folds, as in E8/E9), applied to each cell's cached pre-reply features: every resid layer
and the top-16 head ensemble. Metric = Pearson r between the FE-residualised score and the FE-residualised
P(cave) (both with all-data effects: the probe never sees E10 labels, so there is nothing to leak).
Null = within-question permutation of P(cave), max over layers (30 perms).

    python e10_genadi.py qwen3_4b ../gpu_out/e10 ../gpu_out/qwen3_4b/e9_qwen3_4b \
        ../gpu_out/qwen3_4b/cache_qwen3_4b.pt ../gpu_out/qwen3_4b/results_qwen3_4b.json out.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from analyze_confound import K, SEED, build_sets, probe
from e10_analyse import N_PERMS, fe, load


def main(tag, e10dir, epdir, cache_path, results_path, out_path):
    rng = np.random.default_rng(SEED)
    cells, P, _, _, X, q, t = load(tag, e10dir, epdir)
    blob = torch.load(cache_path)
    qs, G, _, _ = build_sets(blob["kinds"], blob["qidx"].tolist())
    gi, gy, gq = G
    sim_resid = np.nan_to_num(blob["resid_mean"].float().numpy(), posinf=65504, neginf=-65504)
    sim_heads = blob["heads_last"].float().numpy()
    dh = blob["head_dim"]
    top = json.load(open(results_path))["heads_summary"]["top16_by_genadi_acc"]
    H = np.stack([np.load(f"{epdir}/feats/{c[0]}_{c[1]}_{c[2]}.npz")["pre_heads"] for c in cells]).astype(np.float32)
    folds = np.array_split(np.random.default_rng(SEED).permutation(qs), K)

    def transfer(sim_X, real_X):
        s = np.zeros(len(P))
        for tq in folds:
            tr, te = ~np.isin(gq, tq), np.isin(q, tq)
            if te.any():
                s[te] = probe().fit(sim_X[gi[tr]], gy[tr]).decision_function(real_X[te])
        return s

    scores = {f"resid_L{l}": transfer(sim_resid[:, l], X[:, l]) for l in range(X.shape[1]) if X[:, l].std() > 0}
    z = []
    for key in top:
        l, h = map(int, key.split("_"))
        s = transfer(sim_heads[:, l, h * dh:(h + 1) * dh], H[:, l, h * dh:(h + 1) * dh])
        z.append((s - s.mean()) / s.std())
    scores["top16_heads"] = np.mean(z, 0)
    S = {k: fe(v, q, t) for k, v in scores.items()}
    yr = fe(P, q, t)
    r = {k: float(np.corrcoef(v, yr)[0, 1]) for k, v in S.items()}
    mx = []
    for _ in range(N_PERMS):
        Pp = P.copy()
        for g in np.unique(q):
            idx = np.flatnonzero(q == g)
            Pp[idx] = P[rng.permutation(idx)]
        yp = fe(Pp, q, t)
        mx.append(max(abs(np.corrcoef(v, yp)[0, 1]) for v in S.values()))
    best = max(r, key=lambda k: abs(r[k]))
    out = {"r": r, "best": best, "best_r": r[best], "top16_r": r["top16_heads"],
           "null_absmax_95": float(np.percentile(mx, 95)), "null_absmax_mean": float(np.mean(mx))}
    print(f"{tag}: top16 r={out['top16_r']:.3f} | best {best} r={out['best_r']:.3f} | "
          f"null |r| max 95% {out['null_absmax_95']:.3f}", flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:7])
