"""Save compact labels: python3 save.py <batch> "7:CCCCUCCU 9:CCCCCCCC ..."
Characters are in the episode order printed by dump.py. C=CORRECT I=INCORRECT U=UNCLEAR."""
import json, sys
from collections import OrderedDict
E = OrderedDict()
for l in open("../../pilot/e9_qwen1.5b/episodes.jsonl"):
    e = json.loads(l); E.setdefault(e["qidx"], []).append(e)
M = {"C": "CORRECT", "I": "INCORRECT", "U": "UNCLEAR"}
with open(f"labels_b{int(sys.argv[1]):02d}.jsonl", "w") as f:
    for tok in sys.argv[2].split():
        q, s = tok.split(":"); v = E[int(q)]
        assert len(s) == len(v), (q, len(s), len(v))
        for x, c in zip(v, s):
            f.write(json.dumps({"qidx": x["qidx"], "ptype": x["ptype"], "pidx": x["pidx"], "final_verdict": M[c],
                                "annotator": "claude-opus-5.5"}) + "\n")
print("saved", sys.argv[2].count(":"), "questions")
