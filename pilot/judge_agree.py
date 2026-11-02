"""Agreement between the vLLM judge (gpu/judge.py, prompt variants A/B) and Claude's labels.

For each judged set that has Claude labels: Cohen's kappa (3-class C/I/U) for A, B and the A/B consensus
(disagreements -> UNCLEAR), plus kappa on the binary CORRECT vs INCORRECT items both sides committed on.

    python judge_agree.py ../gpu_out/judged
"""
import glob
import json
import os
import sys
from collections import Counter

from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."


def claude_real(path):
    out = {}
    for r in map(json.loads, open(path)):
        out[(r["qidx"], "first")] = r["first_verdict"]
        out[(r["qidx"], "final")] = r["final_verdict"]
    return out


def claude_e9(labdir):
    out = {}
    for f in sorted(glob.glob(f"{labdir}/labels_b*.jsonl")):
        for r in map(json.loads, open(f)):
            out[(r["qidx"], r["ptype"], r["pidx"])] = r["final_verdict"]
    return out


REFS = {
    "calib_real": lambda: claude_real(f"{ROOT}/pilot/real_tqa_qwen1.5b.claude.jsonl"),
    "calib_e9": lambda: claude_e9(f"{ROOT}/handlabels/e9"),
}
for tag in ("qwen3_4b", "llama3b", "gemma4b"):
    p = f"{ROOT}/gpu_out/{tag}/real_tqa_{tag}.claude.jsonl"
    REFS[f"{tag}_real"] = (lambda p=p: claude_real(p))


def key(r):
    return (r["qidx"], r["turn"]) if "turn" in r else (r["qidx"], r["ptype"], r["pidx"])


def main(jdir):
    res = {}
    for name, ref_fn in REFS.items():
        jp = f"{jdir}/judged_{name}.jsonl"
        if not os.path.exists(jp):
            continue
        try:
            ref = ref_fn()
        except FileNotFoundError:
            continue
        rows = [r for r in map(json.loads, open(jp)) if key(r) in ref]
        if not rows:
            continue
        c = [ref[key(r)] for r in rows]
        cons = [r["A"] if r["A"] == r["B"] else "UNCLEAR" for r in rows]
        out = {"n": len(rows), "AB_agree": sum(r["A"] == r["B"] for r in rows) / len(rows)}
        for nm, lab in (("A", [r["A"] for r in rows]), ("B", [r["B"] for r in rows]), ("consensus", cons)):
            out[f"kappa3_{nm}"] = cohen_kappa_score(c, lab)
        both = [(a, b) for a, b in zip(c, cons) if a != "UNCLEAR" and b != "UNCLEAR"]
        out["n_binary"] = len(both)
        out["kappa2_consensus"] = cohen_kappa_score(*zip(*both)) if both else None
        out["confusion_claude_vs_consensus"] = {f"{a[0]}{b[0]}": n for (a, b), n in Counter(zip(c, cons)).items()}
        res[name] = out
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    json.dump(res, open(f"{jdir}/judge_agree.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
