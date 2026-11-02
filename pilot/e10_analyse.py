"""E10 analysis: does the pre-reply state predict P(cave) beyond question and wording main effects?

Frozen before looking at any E10 result (written 2026-11-02 while sampling was running).

Cells = the E9 grid (eligible question x pushback template). For each cell, P(cave) = share of judged
samples (T=0.7, n=10, judge prompt A) that are INCORRECT among those judged CORRECT or INCORRECT;
cells with fewer than MIN_COMMITTED committed samples are dropped. Features = the cached E9 pre-reply
residual stream for that exact prompt (identical tokens).

PRIMARY: two-way fixed effects. Question and template main effects are removed from both P(cave) and
the features (alternating demeaning; template effects estimated on training questions only). A ridge
probe (PCA-256 per layer, label-free) is fit on the residuals with 5 question-disjoint folds; metric =
out-of-fold Pearson r between predicted and actual residual P(cave), per layer. Null = permute P(cave)
across templates within each question (30 perms), whole pipeline re-run; FWER via the max over layers.
GATE: best-layer r > 95th percentile of the null max-over-layers.

Reported alongside:
  ceiling   split-half reliability of the residual P(cave) (samples 0-4 vs 5-9, Spearman-Brown corrected):
            how much interaction variance is real rather than sampling noise.
  q-only    the same probe with only question effects removed (template wording left in): a sanity
            check that the pipeline finds signal when signal is there.

    python e10_analyse.py qwen3_4b ../gpu_out/e10 ../gpu_out/qwen3_4b/e9_qwen3_4b ../gpu_out/e10/e10_qwen3_4b.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SEED, K, MIN_COMMITTED, NPC = 0, 5, 5, 256
N_PERMS = int(os.environ.get("E10_PERMS", 30))  # 30 for the first pass; 200 for the paper
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def load(tag, e10dir, epdir):
    lab = {(r["qidx"], r["ptype"], r["pidx"]): r["labels"]
           for r in map(json.loads, open(f"{e10dir}/{tag}_judged.jsonl"))}
    cells, P, Pa, Pb, X = [], [], [], [], []
    for key, ls in lab.items():
        def rate(sub):
            c = [x for x in sub if x in ("CORRECT", "INCORRECT")]
            return (sum(x == "INCORRECT" for x in c) / len(c)) if c else np.nan, len(c)
        p, n = rate(ls)
        if n < MIN_COMMITTED:
            continue
        cells.append(key)
        P.append(p)
        Pa.append(rate(ls[:len(ls) // 2])[0])
        Pb.append(rate(ls[len(ls) // 2:])[0])
        X.append(np.load(f"{epdir}/feats/{key[0]}_{key[1]}_{key[2]}.npz")["pre_resid"])
    X = np.nan_to_num(np.stack(X).astype(np.float32), posinf=65504, neginf=-65504)
    q = np.array([c[0] for c in cells])
    t = np.array([f"{c[1]}{c[2]}" for c in cells])
    return cells, np.array(P), np.array(Pa), np.array(Pb), X, q, t


def _ind(g):
    """Sparse one-hot (n x groups) and group sizes."""
    from scipy.sparse import csr_matrix
    u, inv = np.unique(g, return_inverse=True)
    return csr_matrix((np.ones(len(g)), (np.arange(len(g)), inv)), shape=(len(g), len(u)))


def fe(y, q, t, tmask=None, it=20):
    """Remove question and template effects from y [n] or [n, d] by alternating demeaning.
    Template effects are estimated only on rows in tmask (training questions) and applied to all rows."""
    tmask = np.ones(len(q), bool) if tmask is None else tmask
    r = y.astype(np.float64).copy()
    one = r.ndim == 1
    if one:
        r = r[:, None]
    Gq, Gt = _ind(q), _ind(t)
    nq = np.asarray(Gq.sum(0)).ravel()[:, None]
    Gt_tr = Gt.multiply(tmask[:, None]).tocsr()
    nt = np.maximum(np.asarray(Gt_tr.sum(0)).ravel()[:, None], 1)
    for _ in range(it):
        r -= Gq @ ((Gq.T @ r) / nq)
        r -= Gt @ ((Gt_tr.T @ r) / nt)
    return r[:, 0] if one else r


def qonly(y, q):
    Gq = _ind(q)
    r = y.astype(np.float64).copy()
    r2 = r[:, None] if r.ndim == 1 else r
    r2 -= Gq @ ((Gq.T @ r2) / np.asarray(Gq.sum(0)).ravel()[:, None])
    return r


def pca(X, k=NPC):
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :k] * S[:k]


def ridge_oof(Z, y, q, t, folds, mode):
    """Out-of-fold predictions and targets; residualisation inside each fold (template effects from the
    training questions only, applied to the test questions; question effects are each question's own)."""
    pred, targ = np.zeros(len(y)), np.zeros(len(y))
    for tq in folds:
        te = np.isin(q, tq)
        tr = ~te
        if mode == "fe":
            yr, Zr = fe(y, q, t, tr), fe(Z, q, t, tr)
        else:
            yr, Zr = qonly(y, q), qonly(Z, q)
        mu, sd = Zr[tr].mean(0), Zr[tr].std(0) + 1e-8
        A, b = (Zr[tr] - mu) / sd, yr[tr]
        best, ba = -np.inf, ALPHAS[0]
        iq = np.unique(q[tr])
        inner = np.isin(q[tr], iq[::2])
        for al in ALPHAS:  # alpha chosen on an inner question split of the training fold
            w = np.linalg.solve(A[~inner].T @ A[~inner] + al * np.eye(A.shape[1]), A[~inner].T @ b[~inner])
            r = np.corrcoef(A[inner] @ w, b[inner])[0, 1]
            if r > best:
                best, ba = r, al
        w = np.linalg.solve(A.T @ A + ba * np.eye(A.shape[1]), A.T @ b)
        pred[te], targ[te] = ((Zr[te] - mu) / sd) @ w, yr[te]
    return pred, targ


def ridge_oof_many(Z, Y, q, t, folds):
    """ridge_oof(..., mode="fe") for many targets at once (columns of Y), returning one oof r per column.
    The feature side (fold-wise residualisation, standardisation, ridge solves) does not depend on y, so it is
    computed once; each target then costs only matrix products. Identical to ridge_oof up to float error."""
    n, m = Y.shape
    pred, targ = np.zeros((n, m)), np.zeros((n, m))
    for tq in folds:
        te = np.isin(q, tq)
        tr = ~te
        Zr, Yr = fe(Z, q, t, tr), fe(Y, q, t, tr)
        mu, sd = Zr[tr].mean(0), Zr[tr].std(0) + 1e-8
        A, B, At = (Zr[tr] - mu) / sd, Yr[tr], (Zr[te] - mu) / sd
        iq = np.unique(q[tr])
        inner = np.isin(q[tr], iq[::2])
        Ain, Aout, I = A[~inner], A[inner], np.eye(A.shape[1])
        best_r, best_k = np.full(m, -np.inf), np.zeros(m, int)
        for k, al in enumerate(ALPHAS):
            P_in = Aout @ np.linalg.solve(Ain.T @ Ain + al * I, Ain.T @ B[~inner])
            b_in = B[inner]
            pc, bc = P_in - P_in.mean(0), b_in - b_in.mean(0)
            r = (pc * bc).sum(0) / np.sqrt((pc ** 2).sum(0) * (bc ** 2).sum(0))
            better = r > best_r  # strict, as in ridge_oof: the first alpha wins ties
            best_r[better], best_k[better] = r[better], k
        for k, al in enumerate(ALPHAS):
            cols = best_k == k
            if cols.any():
                W = np.linalg.solve(A.T @ A + al * I, A.T @ B[:, cols])
                pred[np.ix_(te, cols)] = At @ W
        targ[te] = Yr[te]
    pc, tc = pred - pred.mean(0), targ - targ.mean(0)
    return (pc * tc).sum(0) / np.sqrt((pc ** 2).sum(0) * (tc ** 2).sum(0))


def main(tag, e10dir, epdir, out_path):
    rng = np.random.default_rng(SEED)
    cells, P, Pa, Pb, X, q, t = load(tag, e10dir, epdir)
    L = X.shape[1]
    folds = np.array_split(np.random.default_rng(SEED).permutation(np.unique(q)), K)
    out = {"n_cells": len(P), "n_questions": int(len(np.unique(q))), "n_templates": int(len(np.unique(t))),
           "mean_p_cave": float(P.mean()),
           "p_cave_by_template": {g: float(P[t == g].mean()) for g in np.unique(t)}}

    # variance decomposition + reliability ceiling of the interaction
    tot = P.var()
    rq = qonly(P, q)
    rfe = fe(P, q, t)
    out["var_share"] = {"question": float(1 - rq.var() / tot), "template_given_q": float((rq.var() - rfe.var()) / tot),
                        "interaction_plus_noise": float(rfe.var() / tot)}
    ok = ~np.isnan(Pa) & ~np.isnan(Pb)
    ra, rb = fe(Pa[ok], q[ok], t[ok]), fe(Pb[ok], q[ok], t[ok])
    r_half = float(np.corrcoef(ra, rb)[0, 1])
    out["interaction_reliability"] = {"split_half_r": r_half, "spearman_brown": float(2 * r_half / (1 + r_half))}
    print({k: out[k] for k in ("n_cells", "n_questions", "n_templates", "mean_p_cave")}, flush=True)
    print("variance shares", out["var_share"], "| interaction reliability", out["interaction_reliability"], flush=True)

    Zs = [pca(X[:, l]) for l in range(L)]
    res = {}
    for mode in ("fe", "qonly"):
        rs = []
        for l in range(L):
            pr, yr = ridge_oof(Zs[l], P, q, t, folds, mode)
            rs.append(float(np.corrcoef(pr, yr)[0, 1]))
        res[mode] = rs
        b = int(np.nanargmax(rs))
        print(f"{mode}: best L{b} r={rs[b]:.3f} | L0 {rs[0]:.3f}", flush=True)
    out["r_per_layer"] = res

    # null for the primary (fe): permute P within question, max over layers (same permutation sequence as
    # the original per-permutation loop; computed all at once with ridge_oof_many)
    perms = []
    for _ in range(N_PERMS):
        Pp = P.copy()
        for g in np.unique(q):
            idx = np.flatnonzero(q == g)
            Pp[idx] = P[rng.permutation(idx)]
        perms.append(Pp)
    Y = np.stack([P] + perms, 1)
    R = np.full((N_PERMS + 1, L), np.nan)
    for l in range(L):
        if np.isfinite(res["fe"][l]):
            R[:, l] = ridge_oof_many(Zs[l], Y, q, t, folds)
    assert np.allclose(R[0], res["fe"], equal_nan=True, atol=1e-6), "fast path disagrees with ridge_oof"
    mx = list(np.nanmax(R[1:], 1))
    print(f"null max r: mean {np.mean(mx):.3f}, 95th pct {np.percentile(mx, 95):.3f} ({N_PERMS} perms)", flush=True)
    thr = float(np.percentile(mx, 95))
    b = int(np.nanargmax(res["fe"]))
    out["primary"] = {"best_layer": b, "r": res["fe"][b], "null_max_95": thr, "null_max_mean": float(np.mean(mx)),
                      "p_fwer": float((np.sum(np.array(mx) >= res["fe"][b]) + 1) / (N_PERMS + 1)),
                      "gate_pass": bool(res["fe"][b] > thr)}
    print("PRIMARY", out["primary"], flush=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:5])
