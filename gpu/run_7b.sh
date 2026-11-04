#!/usr/bin/env bash
# Pod side, scale check on Qwen2.5-7B-Instruct (ungated). bf16 throughout: fp32 7B does not fit a 24 GB GPU. Labels come from the
# LLM judge throughout (the paper shows E8/E9 conclusions are unchanged under judge labels).
#   1. simulated dialogues + real episodes (bf16)                         -> $Q/out/qwen7b/
#   2. judge the real episodes (A/B); eligibility = turn-1 consensus CORRECT
#   3. E9 grid, 8 pushbacks, eligible questions only (bf16 features)      -> $Q/e9out/qwen7b/
#   4. judge the E9 finals (A/B); E10: 10 samples per cell (vLLM bf16, T=0.7), judged with prompt A -> $Q/e10/
#   5. touch $Q/DONE (the Mac sync loop copies everything and stops the pod)
set -uo pipefail
cd "$(dirname "$0")"
Q=/workspace/q7; mkdir -p $Q/out $Q/e9out $Q/e10
M=Qwen/Qwen2.5-7B-Instruct; T=qwen7b
log() { echo "[7b] $* $(date +%T)" | tee -a $Q/run.log; }

[ -d /workspace/venv ] || python -m venv --system-site-packages /workspace/venv
[ -d /workspace/vjudge ] || python -m venv /workspace/vjudge
/workspace/venv/bin/pip -q install "transformers>=4.57,<4.58" "datasets<4" "numpy<2.3" accelerate 2>&1 | tail -1
/workspace/vjudge/bin/pip -q install "vllm==0.11.0" "transformers>=4.56,<4.58" 2>&1 | tail -1   # transformers 5.x breaks vllm 0.11
HFP=/workspace/venv/bin/python; VP=/workspace/vjudge/bin/python
$HFP -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))" | tee -a $Q/run.log

log "1. sim + real"
$HFP run_model.py --model $M --tag $T --out $Q/out --stages sim,real --dtype bfloat16 --bs 16 2>&1 | grep --line-buffered -E "^\[(model|sim|real|done)\]|Traceback|Error" | tee -a $Q/run.log

log "2. judge real"
$VP judge.py --out $Q/out --sets ${T}_real 2>&1 | grep --line-buffered -E "^\[judge\]|^\{|Traceback|Error" | tee -a $Q/run.log
$VP - <<PY 2>&1 | tee -a $Q/run.log
import json
J = {}
for l in open("$Q/out/judged/judged_${T}_real.jsonl"):
    r = json.loads(l); J[(r["qidx"], r["turn"])] = r["A"] if r["A"] == r["B"] else "UNCLEAR"
import os; os.makedirs("$Q/e9out/$T", exist_ok=True)
rows = [json.loads(l) for l in open("$Q/out/$T/real_tqa_$T.jsonl")]
with open("$Q/out/$T/real_tqa_$T.judge.jsonl", "w") as a, open("$Q/e9out/$T/real_tqa_$T.jsonl", "w") as b:
    for r in rows:
        r["first_verdict"], r["final_verdict"] = J[(r["qidx"], "first")], J[(r["qidx"], "final")]
        a.write(json.dumps(r, ensure_ascii=False) + "\n")
        if r["first_verdict"] == "CORRECT":
            b.write(json.dumps(r, ensure_ascii=False) + "\n")
print("[7b] eligible", sum(r["first_verdict"] == "CORRECT" for r in rows), "of", len(rows))
PY

log "3. E9 grid"
$HFP run_model.py --model $M --tag $T --out $Q/e9out --stages e9 --npush 8 --dtype bfloat16 --bs 16 2>&1 | grep --line-buffered -E "^\[(model|e9|done)\]|Traceback|Error" | tee -a $Q/run.log

log "4. judge E9 finals, E10 sample + judge"
$VP judge.py --out $Q/e9out --sets ${T}_e9 2>&1 | grep --line-buffered -E "^\[judge\]|^\{|Traceback|Error" | tee -a $Q/run.log
[ -f $Q/e10/${T}_samples.jsonl ] || $VP e10.py sample --model $M --episodes $Q/e9out/$T/e9_$T/episodes.jsonl --dtype bfloat16 \
  --out $Q/e10/${T}_samples.jsonl 2>&1 | grep --line-buffered -E "^\[e10|Traceback|Error" | tee -a $Q/run.log
$VP e10.py judge --samples $Q/e10/${T}_samples.jsonl --out $Q/e10 2>&1 | grep --line-buffered -E "^\[e10|Traceback|Error" | tee -a $Q/run.log

ls $Q/e10/${T}_judged.jsonl $Q/e9out/judged/judged_${T}_e9.jsonl $Q/out/$T/cache_$T.pt >/dev/null 2>&1 && touch $Q/DONE
log "all done (DONE flag: $([ -f $Q/DONE ] && echo yes || echo NO))"
