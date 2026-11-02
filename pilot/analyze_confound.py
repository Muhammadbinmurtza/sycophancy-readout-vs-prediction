"""Analyse the cache from genadi_confound.py.

Sets (per TruthfulQA question q):
    G  (Genadi "sycophancy_challenged"): CI -> 1, and CC (q even) / IC (q odd) -> 0
    T  (truth control, single turn):      T_C -> 1, T_I -> 0
    D  (deconfounded, final answer fixed INCORRECT): CI -> 1 ("caved"), II -> 0 ("stayed wrong")

For every residual layer (mean over final-answer tokens) and every attention head
(o_proj input at the last token) we report:
    acc_genadi   accuracy on G with Genadi's example-level random 80/20 split
    auc_G        AUROC on G with question-level 5-fold CV
    auc_D        probe trained on G (train questions), scored on D (held-out questions)
                 -> ~0.5 if the probe reads "final answer is false" rather than "caved"
    auc_T2G      truth probe trained on T, negated, scored on G (held-out questions)
                 -> high if the "sycophancy" probe is a truth probe in disguise
    cos_GT       cosine of difference-in-means directions, G vs T (expect negative)

    python analyze_confound.py cache_qwen1.5b.pt results_qwen1.5b.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
K = 5


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def build_sets(kinds, qidx):
    idx = {(int(q), k): i for i, (q, k) in enumerate(zip(qidx, kinds))}
    qs = sorted({int(q) for q in qidx})
    G, T, D = [], [], []
    for q in qs:
        G += [(idx[q, "CI"], 1, q), (idx[q, "CC" if q % 2 == 0 else "IC"], 0, q)]
        T += [(idx[q, "T_C"], 1, q), (idx[q, "T_I"], 0, q)]
        D += [(idx[q, "CI"], 1, q), (idx[q, "II"], 0, q)]
    as_arr = lambda s: tuple(np.array(c) for c in zip(*s))
    return qs, as_arr(G), as_arr(T), as_arr(D)


def evaluate(X, qs, G, T, D, folds):
    gi, gy, gq = G
    ti, ty, tq = T
    di, dy, dq = D

    # Genadi protocol: example-level random split, accuracy
    tr, te = train_test_split(np.arange(len(gi)), test_size=0.2, random_state=3407)
    p = probe().fit(X[gi[tr]], gy[tr])
    acc_genadi = float((p.predict(X[gi[te]]) == gy[te]).mean())

    sG, sD, sT2G = np.zeros(len(gi)), np.zeros(len(di)), np.zeros(len(gi))
    for test_q in folds:
        trG, teG = ~np.isin(gq, test_q), np.isin(gq, test_q)
        trT, teD = ~np.isin(tq, test_q), np.isin(dq, test_q)
        pg = probe().fit(X[gi[trG]], gy[trG])
        sG[teG] = pg.decision_function(X[gi[teG]])
        sD[teD] = pg.decision_function(X[di[teD]])
        pt = probe().fit(X[ti[trT]], ty[trT])
        sT2G[teG] = -pt.decision_function(X[gi[teG]])

    dG = X[gi[gy == 1]].mean(0) - X[gi[gy == 0]].mean(0)
    dT = X[ti[ty == 1]].mean(0) - X[ti[ty == 0]].mean(0)
    cos = float(dG @ dT / (np.linalg.norm(dG) * np.linalg.norm(dT) + 1e-12))
    return {"acc_genadi": acc_genadi,
            "auc_G": float(roc_auc_score(gy, sG)),
            "auc_D": float(roc_auc_score(dy, sD)),
            "auc_T2G": float(roc_auc_score(gy, sT2G)),
            "cos_GT": cos}


def main(cache_path, out_path):
    blob = torch.load(cache_path)
    qs, G, T, D = build_sets(blob["kinds"], blob["qidx"].tolist())
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(qs), K)

    resid = np.nan_to_num(blob["resid_mean"].float().numpy(), posinf=65504, neginf=-65504)  # Gemma fp16 overflow
    heads = blob["heads_last"].float().numpy()
    n_heads, d_head = blob["n_heads"], blob["head_dim"]

    res = {"model": blob["model"], "n_questions": len(qs), "resid": {}, "heads": {}}
    for l in range(resid.shape[1]):
        res["resid"][l] = evaluate(resid[:, l], qs, G, T, D, folds)
        r = res["resid"][l]
        print(f"resid L{l:2d}  accG {r['acc_genadi']:.3f}  aucG {r['auc_G']:.3f}  "
              f"aucD {r['auc_D']:.3f}  T->G {r['auc_T2G']:.3f}  cos {r['cos_GT']:+.2f}", flush=True)
    for l in range(heads.shape[1]):
        for h in range(n_heads):
            X = heads[:, l, h * d_head:(h + 1) * d_head]
            res["heads"][f"{l}_{h}"] = evaluate(X, qs, G, T, D, folds)
        print(f"heads layer {l} done", flush=True)

    hv = res["heads"]
    top = sorted(hv, key=lambda k: hv[k]["acc_genadi"], reverse=True)[:16]
    summ = {m: {"top16_mean": float(np.mean([hv[k][m] for k in top])),
                "all_median": float(np.median([hv[k][m] for k in hv]))}
            for m in ("acc_genadi", "auc_G", "auc_D", "auc_T2G", "cos_GT")}
    res["heads_summary"] = {"top16_by_genadi_acc": top, **summ}
    best_l = max(res["resid"], key=lambda l: res["resid"][l]["acc_genadi"])
    res["resid_best_layer"] = {"layer": best_l, **res["resid"][best_l]}
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps({"resid_best_layer": res["resid_best_layer"],
                      "heads_summary": summ}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
