"""E9 analysis: matched within-question test at the PRE-REPLY position (Saad's design) on TruthfulQA.

Frozen before looking at any E9 result (written 2026-11-01 while generation was still running):

PRIMARY (Saad's D14/D15, replicated exactly): label = final INCORRECT (flip, 1) vs final CORRECT
(hold, 0); UNCLEAR excluded. Matched within-question diff-in-means direction, leave-one-question-out,
held-out episodes scored by projection, AUROC pooled over all held-out episodes, per layer of the
pre-reply residual stream. Null = within-question label shuffle (20 perms). GATE: PASS iff best-layer
AUROC >= 0.70 and > shuffle mean + 2sd at that layer.

SECONDARY / controls (reported whatever they show):
  within-q   same LOQO scores, centred per question, AUROC over matched questions only (removes any
             question-level signal that pooled AUROC still contains).
  template   pushback-identity baseline: score = flip rate of that exact pushback prompt in the other
             questions. Measures how much is predictable from prompt surface alone.
  simple     matched LOQO restricted to the 5 'simple' paraphrases (pressure type held fixed).
  resid-tpl  matched LOQO after subtracting per-template feature means (template identity removed).
  genadi     Genadi-style G-probes trained on simulated dialogues (question-disjoint 5 folds, as E8),
             applied at the pre-reply position: best resid layer and top-16 head ensemble.
  answer     the same matched LOQO at the answer read-out, for contrast (read-out, not prediction).
  control    positive control: authoritative vs simple pushback at pre-reply (should be ~1.0).

    python e9_analyse.py e9_qwen1.5b ../handlabels/e9 cache_qwen1.5b.pt results_qwen1.5b.json e9_qwen1.5b.json
    python e9_analyse.py ../gpu_out/<tag>/e9_<tag> ../gpu_out/judged/judged_<tag>_e9.jsonl \\
        ../gpu_out/<tag>/cache_<tag>.pt ../gpu_out/<tag>/results_<tag>.json ../gpu_out/<tag>/e9_<tag>.json
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from analyze_confound import K, SEED, build_sets, probe
from ci_bootstrap import boot_auc

N_PERMS = 20
GATE = 0.70


def load_labels(labsrc):
    """labsrc: a dir of hand labels (labels_b*.jsonl, final_verdict), or a judged_<set>.jsonl from gpu/judge.py,
    where the label is the A/B prompt-variant consensus and disagreements count as UNCLEAR."""
    lab = {}
    if labsrc.endswith(".jsonl"):
        for r in map(json.loads, open(labsrc)):
            lab[(r["qidx"], r["ptype"], r["pidx"])] = r["A"] if r["A"] == r["B"] else "UNCLEAR"
    else:
        for f in sorted(glob.glob(f"{labsrc}/labels_b*.jsonl")):
            for r in map(json.loads, open(f)):
                lab[(r["qidx"], r["ptype"], r["pidx"])] = r["final_verdict"]
    return lab


def load(epdir, labsrc, heads, dh, maxq=None, elig=None):
    """heads: 'L_H' keys to keep from the head features (all heads of a 4B model do not fit in 16 GB).
    elig: real_tqa_<tag>.*.jsonl with first_verdict; keeps only questions whose turn-1 answer is CORRECT
    (the GPU run generated pushbacks for all 817 questions; eligibility is applied here)."""
    lab = load_labels(labsrc)
    ok = None
    if elig:
        ok = {r["qidx"] for r in map(json.loads, open(elig)) if r["first_verdict"] == "CORRECT"}
    eps = [e for e in map(json.loads, open(f"{epdir}/episodes.jsonl"))
           if lab.get((e["qidx"], e["ptype"], e["pidx"])) in ("CORRECT", "INCORRECT")
           and (ok is None or e["qidx"] in ok)]
    if maxq:  # smoke tests only
        keepq = sorted({e["qidx"] for e in eps})[:maxq]
        eps = [e for e in eps if e["qidx"] in keepq]
    y = np.array([lab[(e["qidx"], e["ptype"], e["pidx"])] == "INCORRECT" for e in eps], int)
    lh = [tuple(map(int, k.split("_"))) for k in heads]
    feats = {k: [] for k in ("pre_resid", "pre_heads", "ans_resid", "ans_heads")}
    for e in eps:
        z = np.load(f"{epdir}/feats/{e['qidx']}_{e['ptype']}_{e['pidx']}.npz")
        for k in feats:
            v = z[k]
            feats[k].append(np.stack([v[l, h * dh:(h + 1) * dh] for l, h in lh]) if k.endswith("heads") else v)
    feats = {k: np.nan_to_num(np.stack(v).astype(np.float32), posinf=65504, neginf=-65504) for k, v in feats.items()}
    return eps, y, feats, len(lab)


def matched_loqo(X, y, q):
    """X [n, L, d]. Saad's matched direction, LOQO. Returns scores [n, L]."""
    uq = np.unique(q)
    D, has = {}, []
    for g in uq:
        m = q == g
        if 0 < y[m].sum() < m.sum():
            D[g] = X[m & (y == 1)].mean(0) - X[m & (y == 0)].mean(0)
    tot = sum(D.values())
    n = len(D)
    s = np.full(X.shape[:2], np.nan, np.float32)
    for g in uq:
        v = (tot - D[g]) / (n - 1) if g in D else tot / n
        v = v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-8)
        m = q == g
        s[m] = np.einsum("nld,ld->nl", X[m], v)
    return s, n


def pooled_auc(y, s):
    return np.array([roc_auc_score(y, s[:, l]) for l in range(s.shape[1])])


def withinq_auc(y, s, q):
    """Centre scores per question; AUROC over questions that have both labels."""
    keep = np.zeros(len(y), bool)
    c = s.copy()
    for g in np.unique(q):
        m = q == g
        c[m] -= c[m].mean(0)
        keep |= m & (0 < y[m].sum() < m.sum())
    return pooled_auc(y[keep], c[keep]), keep


def shuffle_null(X, y, q, rng):
    out = []
    for _ in range(N_PERMS):
        ys = y.copy()
        for g in np.unique(q):
            idx = np.flatnonzero(q == g)
            ys[idx] = y[rng.permutation(idx)]
        s, _ = matched_loqo(X, ys, q)
        out.append(pooled_auc(ys, s))
    out = np.array(out)
    return out.mean(0), out.std(0)


def curve(X, y, q, rng, null=True):
    s, n = matched_loqo(X, y, q)
    auc = pooled_auc(y, s)
    wq, keep = withinq_auc(y, s, q)
    res = {"n_matched_q": int(n), "pooled": auc.tolist(), "withinq": wq.tolist()}
    b = int(auc.argmax())
    res["best_layer"] = b
    res["best_pooled_ci"] = boot_auc(y, s[:, b], q, rng)
    if null:
        mu, sd = shuffle_null(X, y, q, rng)
        res["null_mean"], res["null_sd"] = mu.tolist(), sd.tolist()
        res["gate_pass"] = bool(auc[b] >= GATE and auc[b] > mu[b] + 2 * sd[b])
    return res, s


def main(epdir, labsrc, cache_path, results_path, out_path, maxq=None, elig=None):
    rng = np.random.default_rng(SEED)
    blob = torch.load(cache_path)
    dh = blob["head_dim"]
    top = json.load(open(results_path))["heads_summary"]["top16_by_genadi_acc"]
    eps, y, F, n_lab = load(epdir, labsrc, top, dh, int(maxq) if maxq else None, elig)
    q = np.array([e["qidx"] for e in eps])
    tpl = np.array([f"{e['ptype']}{e['pidx']}" for e in eps])
    ptype = np.array([e["ptype"] for e in eps])
    out = {"n_labelled": n_lab, "n_used": len(y), "n_flip": int(y.sum()), "n_hold": int((1 - y).sum()),
           "n_questions": int(len(np.unique(q))),
           "flip_rate_by_type": {t: float(y[ptype == t].mean()) for t in np.unique(ptype)}}
    print({k: out[k] for k in ("n_used", "n_flip", "n_hold", "n_questions")}, flush=True)
    print("flip rate by type", out["flip_rate_by_type"], flush=True)

    # PRIMARY
    out["pre_resid"], s_pre = curve(F["pre_resid"], y, q, rng)
    r = out["pre_resid"]
    b = r["best_layer"]
    print(f"PRIMARY pre-reply resid: best L{b} pooled {r['pooled'][b]:.3f} {r['best_pooled_ci']['ci95']} "
          f"null {r['null_mean'][b]:.3f}+-{r['null_sd'][b]:.3f} gate {'PASS' if r['gate_pass'] else 'FAIL'} "
          f"| L0 {r['pooled'][0]:.3f} | within-q best {max(r['withinq']):.3f}", flush=True)

    # template-identity baseline
    st = np.zeros(len(y))
    for g in np.unique(q):
        tr, te = q != g, q == g
        rate = {t: y[tr & (tpl == t)].mean() for t in np.unique(tpl[te])}
        st[te] = [rate[t] for t in tpl[te]]
    wq, keep = withinq_auc(y, st[:, None], q)
    out["template_baseline"] = {"pooled": boot_auc(y, st, q, rng), "withinq": float(wq[0])}
    print("template baseline", out["template_baseline"], flush=True)

    # simple-only (pressure type held fixed)
    # (needs >1 simple paraphrase per question, i.e. the 8-pushback design; skipped when npush=4)
    m = ptype == "simple"
    if len(np.unique(tpl[m])) > 1:
        out["pre_resid_simple"], _ = curve(F["pre_resid"][m], y[m], q[m], rng)
        r = out["pre_resid_simple"]
        print(f"simple-only: best L{r['best_layer']} {max(r['pooled']):.3f} "
              f"gate {'PASS' if r['gate_pass'] else 'FAIL'}", flush=True)
    else:
        print("simple-only: skipped (one simple paraphrase per question)", flush=True)

    # template-residualised
    Xr = F["pre_resid"].copy()
    for t in np.unique(tpl):
        Xr[tpl == t] -= Xr[tpl == t].mean(0)
    out["pre_resid_tplres"], _ = curve(Xr, y, q, rng)
    r = out["pre_resid_tplres"]
    print(f"template-residualised: best L{r['best_layer']} {max(r['pooled']):.3f} "
          f"gate {'PASS' if r['gate_pass'] else 'FAIL'}", flush=True)
    del Xr

    # answer read-out, for contrast
    out["ans_resid"], _ = curve(F["ans_resid"], y, q, rng, null=False)
    r = out["ans_resid"]
    print(f"answer read-out: best L{r['best_layer']} {max(r['pooled']):.3f}", flush=True)

    # positive control: authoritative vs simple, pre-reply
    m = np.isin(ptype, ["simple", "authoritative"])
    yc = (ptype[m] == "authoritative").astype(int)
    sc, _ = matched_loqo(F["pre_resid"][m], yc, q[m])
    out["control_auth_vs_simple"] = float(pooled_auc(yc, sc).max())
    print("positive control", out["control_auth_vs_simple"], flush=True)

    # Genadi-style G-probes transferred to the pre-reply position
    qs, G, _, _ = build_sets(blob["kinds"], blob["qidx"].tolist())
    gi, gy, gq = G
    sim_resid = np.nan_to_num(blob["resid_mean"].float().numpy(), posinf=65504, neginf=-65504)  # Gemma fp16 overflow
    sim_heads = blob["heads_last"].float().numpy()
    folds = np.array_split(np.random.default_rng(SEED).permutation(qs), K)

    def transfer(sim_X, real_X):
        s = np.zeros(len(y))
        for tq in folds:
            tr, te = ~np.isin(gq, tq), np.isin(q, tq)
            if te.any():
                s[te] = probe().fit(sim_X[gi[tr]], gy[tr]).decision_function(real_X[te])
        return s

    g = {}
    for ro in ("pre", "ans"):
        R, H = F[f"{ro}_resid"], F[f"{ro}_heads"]
        per = [roc_auc_score(y, transfer(sim_resid[:, l], R[:, l])) for l in range(R.shape[1])]
        z = []
        for i, key in enumerate(top):
            l, h = map(int, key.split("_"))
            s = transfer(sim_heads[:, l, h * dh:(h + 1) * dh], H[:, i])
            z.append((s - s.mean()) / s.std())
        s = np.mean(z, 0)
        wq, _ = withinq_auc(y, s[:, None], q)
        g[ro] = {"resid_per_layer": per, "top16": boot_auc(y, s, q, rng), "top16_withinq": float(wq[0])}
        print(f"genadi transfer {ro}: top16 {g[ro]['top16']} within-q {wq[0]:.3f} | "
              f"resid range [{min(per):.3f}, {max(per):.3f}]", flush=True)
    out["genadi_transfer"] = g
    json.dump(out, open(out_path, "w"), indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:8])
