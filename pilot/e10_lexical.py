"""E10 baseline: can surface text alone predict the question x pushback interaction in P(cave)?

Same cells, targets and two-way FE ridge pipeline as e10_analyse.py, but the features are text, not
activations: TF-IDF of (question + the model's first answer), reduced to 64 dims, crossed with a one-hot of
the pushback template (64 x n_templates interaction features). After question and template effects are
removed, this is the strongest purely lexical account of "this pushback on this question". If it matches
the activation probe, the E10 signal is surface text; if it is far below, the activations carry more.

    python e10_lexical.py qwen3_4b ../gpu_out/e10 ../gpu_out/qwen3_4b/e9_qwen3_4b out.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from e10_analyse import K, SEED, load, ridge_oof


def main(tag, e10dir, epdir, out_path):
    cells, P, _, _, _, q, t = load(tag, e10dir, epdir)
    ep = {(e["qidx"], e["ptype"], e["pidx"]): e for e in map(json.loads, open(f"{epdir}/episodes.jsonl"))}
    text = [ep[c]["question"] + " " + ep[c]["first"] for c in cells]
    E = TruncatedSVD(64, random_state=SEED).fit_transform(TfidfVectorizer(min_df=2).fit_transform(text))
    tu = sorted(set(t))
    T = np.array([[ti == u for u in tu] for ti in t], float)
    Z = (E[:, :, None] * T[:, None, :]).reshape(len(t), -1)
    folds = np.array_split(np.random.default_rng(SEED).permutation(np.unique(q)), K)
    pr, yr = ridge_oof(Z, P, q, t, folds, "fe")
    r = float(np.corrcoef(pr, yr)[0, 1])
    print(f"{tag}: lexical (question+first TF-IDF x template) two-way FE oof r = {r:.3f}", flush=True)
    json.dump({"r_lexical": r, "n_features": int(Z.shape[1])}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
