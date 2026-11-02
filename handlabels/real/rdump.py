"""Print real (2-turn) episodes for annotation. python3 rdump.py <tag> <batch> [per_batch]
Each question gets a 2-char label: turn-1 verdict + final verdict (C=CORRECT, I=INCORRECT, U=UNCLEAR)."""
import json, sys
tag, b = sys.argv[1], int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 25
rows = [json.loads(l) for l in open(f"../../gpu_out/{tag}/real_tqa_{tag}.jsonl")][b * n:(b + 1) * n]
for r in rows:
    print(f"### Q{r['qidx']}: {r['question']}\n  TRUE: {' | '.join(r['correct_answers'][:3])}\n  FALSE: {' | '.join(r['incorrect_answers'][:3])}")
    print(f"  [1] {' '.join(r['first'].split())[:300]}\n  [2] {' '.join(r['final'].split())[:300]}")
