#!/usr/bin/env bash
# Pod side: E10 sampling (Qwen2.5-1.5B, Qwen3-4B) then judging; touches E10_DONE for the Mac sync loop.
# Inputs uploaded to /workspace/e10in: e9_qwen1.5b_episodes.jsonl, e9_qwen3_4b_episodes.jsonl,
# real_tqa_qwen3_4b.claude.jsonl. The Qwen-1.5B episodes are already restricted to eligible questions.
set -uo pipefail
cd "$(dirname "$0")"
IN=/workspace/e10in; OUT=/workspace/e10; mkdir -p "$OUT"
for v in HF_TOKEN; do [ -n "${!v:-}" ] || export "$(tr '\0' '\n' < /proc/1/environ | grep "^$v=" || true)"; done
[ -d /workspace/vjudge ] || python -m venv /workspace/vjudge
source /workspace/vjudge/bin/activate
pip -q install "vllm==0.11.0" "transformers>=4.56,<4.58" 2>&1 | tail -2   # transformers 5.x breaks vllm 0.11 tokenizers
python -c "import vllm, transformers; print('vllm', vllm.__version__, 'transformers', transformers.__version__)"
# bf16: vLLM fp32 on the 4090 stalled (>30 min, no output) on 2026-11-02; P(cave) is a sampled estimate, so bf16 is fine.
samp() { # tag model episodes [extra args]
  local tag=$1 model=$2 eps=$3; shift 3
  for dt in bfloat16; do
    [ -f $OUT/${tag}_samples.jsonl ] && break
    python e10.py sample --model $model --episodes $eps --dtype $dt --out $OUT/${tag}_samples.jsonl "$@" \
      2>&1 | grep --line-buffered -v -E "it/s|INFO" | tee -a $OUT/e10.log
  done
}
samp qwen1.5b Qwen/Qwen2.5-1.5B-Instruct $IN/e9_qwen1.5b_episodes.jsonl
samp qwen3_4b Qwen/Qwen3-4B-Instruct-2507 $IN/e9_qwen3_4b_episodes.jsonl --elig $IN/real_tqa_qwen3_4b.claude.jsonl
python e10.py judge --samples $OUT/qwen1.5b_samples.jsonl $OUT/qwen3_4b_samples.jsonl --out $OUT \
    2>&1 | grep --line-buffered -v -E "it/s|INFO" | tee -a $OUT/e10.log
ls $OUT/*_judged.jsonl >/dev/null 2>&1 && touch $OUT/E10_DONE
echo "[e10] all done $(date)" | tee -a $OUT/e10.log
