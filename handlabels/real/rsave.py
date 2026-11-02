"""python3 rsave.py <tag> <batch> "0:CI 1:CC ..."  -> <tag>/labels_bXX.jsonl"""
import json, os, sys
tag, b, s = sys.argv[1], int(sys.argv[2]), sys.argv[3]
M = {"C": "CORRECT", "I": "INCORRECT", "U": "UNCLEAR"}
os.makedirs(tag, exist_ok=True)
with open(f"{tag}/labels_b{b:02d}.jsonl", "w") as f:
    for tok in s.split():
        q, v = tok.split(":"); assert len(v) == 2, tok
        f.write(json.dumps({"qidx": int(q), "first_verdict": M[v[0]], "final_verdict": M[v[1]], "annotator": "claude-opus-5.5"}) + "\n")
print("saved", len(s.split()))
