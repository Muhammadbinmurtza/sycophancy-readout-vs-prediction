"""2x2 decomposition of what a Genadi-style probe responds to.

Every TruthfulQA question has all four teacher-forced dialogues
    first answer in {Correct, Incorrect} x final answer in {Correct, Incorrect}
    CI, CC, IC, II
A probe trained on Genadi's labels (G set) is scored, held-out by question (5-fold), on
all four. With per-kind mean scores s, in units of the pooled within-kind SD:
    final_false = ((CI + II) - (CC + IC)) / 2   main effect of the FINAL answer being false
    first_true  = ((CI + CC) - (II + IC)) / 2   main effect of the FIRST answer being true
    changed     = ((CI + IC) - (CC + II)) / 2   the answer changed (= first x final interaction)
"Sycophancy" in Genadi's sense (correct -> incorrect) is the CI cell; a probe that only
reads the final answer's truth value has final_false >> first_true, changed.

    python factorial_confound.py cache_qwen1.5b.pt results_qwen1.5b.json factorial_qwen1.5b.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from analyze_confound import K, SEED, build_sets, probe

CELLS = ("CI", "CC", "IC", "II")


def decompose(X, kinds, qidx, G, folds):
    gi, gy, gq = G
    scores = np.full(len(kinds), np.nan)
    for test_q in folds:
        tr = ~np.isin(gq, test_q)
        p = probe().fit(X[gi[tr]], gy[tr])
        te = np.isin(qidx, test_q) & np.isin(kinds, CELLS)
        scores[te] = p.decision_function(X[te])
    m = {c: scores[kinds == c] for c in CELLS}
    sd = np.sqrt(np.mean([m[c].var() for c in CELLS]))
    mu = {c: float(m[c].mean() / sd) for c in CELLS}
    return {"cell_means_sd_units": mu,
            "final_false": (mu["CI"] + mu["II"] - mu["CC"] - mu["IC"]) / 2,
            "first_true": (mu["CI"] + mu["CC"] - mu["II"] - mu["IC"]) / 2,
            "changed": (mu["CI"] + mu["IC"] - mu["CC"] - mu["II"]) / 2}


def main(cache_path, results_path, out_path):
    blob = torch.load(cache_path)
    kinds = np.array(blob["kinds"])
    qidx = blob["qidx"].numpy()
    qs, G, _, _ = build_sets(blob["kinds"], qidx.tolist())
    folds = np.array_split(np.random.default_rng(SEED).permutation(qs), K)
    resid = np.nan_to_num(blob["resid_mean"].float().numpy(), posinf=65504, neginf=-65504)  # Gemma fp16 overflow
    heads = blob["heads_last"].float().numpy()
    d_head = blob["head_dim"]
    top = json.load(open(results_path))["heads_summary"]["top16_by_genadi_acc"]

    out = {"resid": {}, "top_heads": {}}
    for l in range(resid.shape[1]):
        r = out["resid"][l] = decompose(resid[:, l], kinds, qidx, G, folds)
        print(f"resid L{l:2d}  final_false {r['final_false']:+.2f}  first_true {r['first_true']:+.2f}  "
              f"changed {r['changed']:+.2f}", flush=True)
    for key in top:
        l, h = map(int, key.split("_"))
        r = out["top_heads"][key] = decompose(heads[:, l, h * d_head:(h + 1) * d_head], kinds, qidx, G, folds)
        print(f"head {key:>6}  final_false {r['final_false']:+.2f}  first_true {r['first_true']:+.2f}  "
              f"changed {r['changed']:+.2f}", flush=True)
    out["top_heads_mean"] = {k: float(np.mean([out["top_heads"][h][k] for h in top]))
                             for k in ("final_false", "first_true", "changed")}
    print("top-16 heads mean:", out["top_heads_mean"])
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
