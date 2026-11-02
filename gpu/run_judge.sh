#!/usr/bin/env bash
# Pod side: wait for generation (ALL_DONE), free disk, install vLLM in its own venv, run the judge -> JUDGE_DONE.
set -uo pipefail
OUT=/workspace/out
until [ -f $OUT/ALL_DONE ]; do sleep 60; done
pkill -f "sleep 3600" || true   # cancel run_all.sh's (non-working) auto-stop wait
rm -rf /root/.cache/huggingface/hub/models--meta-llama--* /root/.cache/huggingface/hub/models--google--* \
       /root/.cache/huggingface/hub/models--Qwen--Qwen3-*
df -h / | tail -1
[ -d /workspace/vjudge ] || python -m venv /workspace/vjudge
source /workspace/vjudge/bin/activate
pip -q install "vllm==0.11.0" "transformers>=4.56,<4.58" 2>&1 | tail -2   # transformers 5.x breaks vllm 0.11 tokenizers
python /workspace/code/judge.py --out $OUT --calib /workspace/calib 2>&1 | grep -v -E "it/s|INFO" | tee -a $OUT/judge.log
