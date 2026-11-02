#!/usr/bin/env bash
# Pod side, E10 on two more families (Llama-3.2-3B, Gemma-3-4B). HF_TOKEN must be a pod env var (gated models).
# 1) Extend each model's E9 grid from 4 to 8 pushbacks (adds simple paraphrases 1-4) for eligible questions only,
#    with fp32 pre-reply features, exactly as run_model.py made the first four (existing episodes are skipped).
# 2) Sample 10 replies per cell (vLLM bf16, T=0.7) and judge them (prompt A) -> E10B_DONE.
# Inputs in /workspace/out/<tag>/: real_tqa_<tag>.jsonl (eligible rows only) and e9_<tag>/episodes.jsonl (the
# existing 4-pushback episodes), plus /workspace/e10in/real_tqa_<tag>.claude.jsonl for eligibility.
set -uo pipefail
cd "$(dirname "$0")"
OUT=/workspace/out; E10=/workspace/e10; IN=/workspace/e10in; mkdir -p "$E10"
for v in HF_TOKEN; do [ -n "${!v:-}" ] || export "$(tr '\0' '\n' < /proc/1/environ | grep "^$v=" || true)"; done
[ -n "${HF_TOKEN:-}" ] || { echo "[e10b] HF_TOKEN missing (needed for Llama/Gemma)" | tee -a $E10/e10b.log; exit 1; }

# --- 1. E9 grid extension (HF transformers, fp32) ---
[ -d /workspace/venv ] || python -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
pip -q install "transformers>=4.57,<4.58" "datasets<4" "numpy<2.3" accelerate 2>&1 | tail -1
python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
for spec in "meta-llama/Llama-3.2-3B-Instruct llama3b" "google/gemma-3-4b-it gemma4b"; do
  set -- $spec
  python run_model.py --model "$1" --tag "$2" --out $OUT --stages e9 --npush 8 2>&1 \
    | grep --line-buffered -E "^\[e9\] [0-9]+ episodes|^\[done\]|Traceback|Error" | tee -a $E10/e10b.log
done
deactivate

# --- 2. E10 sampling + judge (vLLM) ---
[ -d /workspace/vjudge ] || python -m venv /workspace/vjudge
source /workspace/vjudge/bin/activate
pip -q install "vllm==0.11.0" "transformers>=4.56,<4.58" 2>&1 | tail -2
python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
for spec in "meta-llama/Llama-3.2-3B-Instruct llama3b" "google/gemma-3-4b-it gemma4b"; do
  set -- $spec
  [ -f $E10/${2}_samples.jsonl ] || python e10.py sample --model "$1" --episodes $OUT/$2/e9_$2/episodes.jsonl \
    --elig $IN/real_tqa_$2.claude.jsonl --dtype bfloat16 --out $E10/${2}_samples.jsonl 2>&1 \
    | grep --line-buffered -v -E "it/s|INFO" | tee -a $E10/e10b.log
done
python e10.py judge --samples $E10/llama3b_samples.jsonl $E10/gemma4b_samples.jsonl --out $E10 2>&1 \
  | grep --line-buffered -v -E "it/s|INFO" | tee -a $E10/e10b.log
ls $E10/llama3b_judged.jsonl $E10/gemma4b_judged.jsonl >/dev/null 2>&1 && touch $E10/E10B_DONE
echo "[e10b] all done $(date)" | tee -a $E10/e10b.log
