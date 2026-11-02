"""Generate REAL pushback episodes on TruthfulQA with Genadi et al.'s exact prompts, so that
simulated-vs-real can be compared without a dataset shift (pilot step iii).

Turn 1: system + question (+ Genadi suffix) -> greedy answer.
Turn 2: Genadi pushback (+ suffix) -> greedy answer.
Resumable: appends one JSON line per question to --out and skips questions already present.
Labelling (correct/incorrect for each turn) is done separately by a judge.

    python generate_real_tqa.py --model Qwen/Qwen2.5-1.5B-Instruct --out real_tqa_qwen1.5b.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from genadi_confound import PUSHBACK, SUFFIX, SYSTEM


@torch.no_grad()
def reply(model, tok, msgs, max_new_tokens):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to("mps").eval()
    ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]

    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["qidx"] for l in open(args.out)}
    with open(args.out, "a") as f:
        for qi, row in enumerate(tqdm(ds, desc="episodes")):
            if qi in done:
                continue
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": row["question"] + SUFFIX}]
            first = reply(model, tok, msgs, args.max_new_tokens)
            msgs += [{"role": "assistant", "content": first},
                     {"role": "user", "content": PUSHBACK}]
            final = reply(model, tok, msgs, args.max_new_tokens)
            f.write(json.dumps({"qidx": qi, "question": row["question"],
                                "correct_answers": row["correct_answers"],
                                "incorrect_answers": row["incorrect_answers"],
                                "best_answer": row["best_answer"],
                                "first": first, "final": final}, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
