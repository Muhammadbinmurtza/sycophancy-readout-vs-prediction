"""Wording-only baseline inside the five simple paraphrases (E9): score = flip rate of that exact paraphrase on the
other questions; within-question AUROC (scores centred per question) over questions with both a flip and a hold.
    python e9_template_simple.py <episodes.jsonl> <judged_e9.jsonl> [elig real_tqa.*.jsonl]"""
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def main(ep, judged, elig=None):
    lab = {}
    for r in map(json.loads, open(judged)):
        lab[(r["qidx"], r["ptype"], r["pidx"])] = r["A"] if r["A"] == r["B"] else "UNCLEAR"
    ok = {r["qidx"] for r in map(json.loads, open(elig)) if r["first_verdict"] == "CORRECT"} if elig else None
    rows = []
    for e in map(json.loads, open(ep)):
        k = (e["qidx"], e["ptype"], e["pidx"])
        if e["ptype"] != "simple" or lab.get(k) not in ("CORRECT", "INCORRECT") or (ok and e["qidx"] not in ok):
            continue
        rows.append((e["qidx"], e["pidx"], int(lab[k] == "INCORRECT")))
    byq = defaultdict(list)
    for r in rows:
        byq[r[0]].append(r)
    matched = [q for q, rs in byq.items() if 0 < sum(r[2] for r in rs) < len(rs)]
    s, y = [], []
    for q in matched:
        sc = []
        for _, p, f in byq[q]:
            other = [r[2] for r in rows if r[1] == p and r[0] != q]
            sc.append(np.mean(other))
        sc = np.array(sc) - np.mean(sc)
        s += list(sc); y += [r[2] for r in byq[q]]
    print(json.dumps({"n_matched_q": len(matched), "withinq_auc": roc_auc_score(y, s),
                      "flip_rate_by_paraphrase": {p: float(np.mean([r[2] for r in rows if r[1] == p])) for p in range(5)}}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
