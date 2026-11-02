"""E8: simulated vs REAL on the SAME dataset (TruthfulQA) and the SAME prompts (Genadi's).
Removes the TriviaQA dataset shift of E6.

Step 1 (extract, needs GPU/MPS, no labels):
    python e8_real_tqa.py extract real_tqa_qwen1.5b.jsonl e8_feats_qwen1.5b.npz
Step 2 (analyse, CPU, needs judged labels):
    python e8_real_tqa.py analyse cache_qwen1.5b.pt results_qwen1.5b.json \
        real_tqa_qwen1.5b.judged.jsonl e8_feats_qwen1.5b.npz e8_qwen1.5b.json

Real labels (judge on each turn): real_CI = first CORRECT, final INCORRECT (caved);
real_CC = both CORRECT (held). Question-disjointness: the G-probe is trained on the simulated
dialogues of the questions in the training folds only, and scored on real episodes of held-out
questions, so no TruthfulQA question is seen in both roles.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from genadi_confound import PUSHBACK, SUFFIX, SYSTEM


def extract(inp, out, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to("mps").eval()
    layers = model.model.layers
    L = len(layers)
    z = {}
    hooks = [layers[i].self_attn.o_proj.register_forward_pre_hook(
        (lambda i: lambda m, inp_: z.__setitem__(i, inp_[0][0].detach()))(i)) for i in range(L)]
    rows = [json.loads(l) for l in open(inp)]
    feats = {k: [] for k in ("ans_resid", "ans_heads", "pre_resid", "pre_heads")}
    for r in rows:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["question"] + SUFFIX},
                {"role": "assistant", "content": r["first"]},
                {"role": "user", "content": PUSHBACK}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": r["final"]}],
                                       tokenize=False, add_generation_prompt=False)
        assert full.startswith(prompt)
        ids = tok(full, add_special_tokens=False)["input_ids"]
        p = len(tok(prompt, add_special_tokens=False)["input_ids"])
        a = p + len(tok(r["final"], add_special_tokens=False)["input_ids"])
        with torch.no_grad():
            o = model(torch.tensor([ids], device="mps"), output_hidden_states=True)
        hs = torch.stack(o.hidden_states)[:, 0]
        heads = torch.stack([z[i] for i in range(L)])
        feats["ans_resid"].append(hs[:, p:a].mean(1).half().cpu())
        feats["ans_heads"].append(heads[:, -1].half().cpu())
        feats["pre_resid"].append(hs[:, p - 1].half().cpu())
        feats["pre_heads"].append(heads[:, p - 1].half().cpu())
    for h in hooks:
        h.remove()
    np.savez_compressed(out, qidx=np.array([r["qidx"] for r in rows]),
                        **{k: torch.stack(v).numpy() for k, v in feats.items()})
    print("saved", out, len(rows))


def analyse(cache_path, results_path, judged_path, feats_path, out_path):
    from analyze_confound import K, SEED, build_sets, probe
    from ci_bootstrap import boot_auc

    rng = np.random.default_rng(SEED)
    blob = torch.load(cache_path)
    qs, G, _, _ = build_sets(blob["kinds"], blob["qidx"].tolist())
    gi, gy, gq = G
    sim_resid = np.nan_to_num(blob["resid_mean"].float().numpy(), posinf=65504, neginf=-65504)  # Gemma fp16 overflow
    sim_heads = blob["heads_last"].float().numpy()
    dh = blob["head_dim"]
    top = json.load(open(results_path))["heads_summary"]["top16_by_genadi_acc"]

    verdict = {}
    for l in open(judged_path):
        r = json.loads(l)
        verdict[r["qidx"]] = (r["first_verdict"], r["final_verdict"])
    d = np.load(feats_path)
    q = d["qidx"]
    kind = np.array([{("CORRECT", "INCORRECT"): "CI", ("CORRECT", "CORRECT"): "CC"}
                     .get(verdict.get(int(x)), "other") for x in q])
    m = np.isin(kind, ["CI", "CC"])
    y, qq = (kind[m] == "CI").astype(int), q[m]
    print({k: int((kind == k).sum()) for k in ("CI", "CC", "other")}, flush=True)

    folds = np.array_split(np.random.default_rng(SEED).permutation(qs), K)

    def transfer(sim_X, real_X):
        s = np.zeros(len(y))
        for tq in folds:
            tr, te = ~np.isin(gq, tq), np.isin(qq, tq)
            if te.any():
                s[te] = probe().fit(sim_X[gi[tr]], gy[tr]).decision_function(real_X[te])
        return s

    def indomain(real_X):
        s = np.zeros(len(y))
        for tq in folds:
            te = np.isin(qq, tq)
            if te.any() and (~te).any():
                s[te] = probe().fit(real_X[~te], y[~te]).decision_function(real_X[te])
        return s

    res = {"counts": {k: int((kind == k).sum()) for k in ("CI", "CC", "other")}, "per_layer": {}}
    for ro in ("ans", "pre"):
        R = np.nan_to_num(d[f"{ro}_resid"][m].astype(np.float32), posinf=65504, neginf=-65504)
        for l in range(R.shape[1]):
            res["per_layer"][f"{ro}_L{l}"] = {
                "transfer": float(boot_auc(y, transfer(sim_resid[:, l], R[:, l]), qq, rng)["auc"]),
                "indomain": float(boot_auc(y, indomain(R[:, l]), qq, rng)["auc"])}
        Hs = d[f"{ro}_heads"][m].astype(np.float32)
        tz, iz = [], []
        for key in top:
            l, h = map(int, key.split("_"))
            sl = slice(h * dh, (h + 1) * dh)
            for lst, s in ((tz, transfer(sim_heads[:, l, sl], Hs[:, l, sl])), (iz, indomain(Hs[:, l, sl]))):
                lst.append((s - s.mean()) / s.std())
        res[f"{ro}_top16_transfer"] = boot_auc(y, np.mean(tz, 0), qq, rng)
        res[f"{ro}_top16_indomain"] = boot_auc(y, np.mean(iz, 0), qq, rng)
        for k in ("transfer", "indomain"):
            v = res[f"{ro}_top16_{k}"]
            print(f"{ro} top16 {k:9s} AUROC {v['auc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    {"extract": extract, "analyse": analyse}[sys.argv[1]](*sys.argv[2:])
