#!/usr/bin/env bash
# Runs on the RunPod pod. HF_TOKEN must be set as a pod environment variable (RunPod UI), never pasted in chat.
#   bash run_all.sh test   # 20-question timing test on Llama-3.2-3B (~10 min)
#   bash run_all.sh full   # all three models, then auto-stop the pod after a 60-min grace period for syncing
set -euo pipefail
cd "$(dirname "$0")"
OUT=/workspace/out; mkdir -p "$OUT"
# RunPod puts pod env vars on PID 1 only; SSH sessions don't inherit them.
for v in HF_TOKEN RUNPOD_POD_ID RUNPOD_API_KEY; do
  [ -n "${!v:-}" ] || export "$(tr '\0' '\n' < /proc/1/environ | grep "^$v=" || true)"
done
# Ubuntu 24.04 blocks system pip (PEP 668): use a venv that still sees the template's torch.
[ -d /workspace/venv ] || python -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
pip -q install "transformers>=4.57,<4.58" "datasets<4" "numpy<2.3" accelerate
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
python - <<'PY'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
PY
case "${1:-test}" in
  test)
    python run_model.py --model meta-llama/Llama-3.2-3B-Instruct --tag llama3b --out "$OUT" --limit 20 2>&1 | tee "$OUT/test.log"
    python run_model.py --model google/gemma-3-4b-it --tag gemma4b --out "$OUT" --limit 5 2>&1 | tee -a "$OUT/test.log" ;;
  full)
    for spec in "Qwen/Qwen3-4B-Instruct-2507 qwen3_4b 8" "meta-llama/Llama-3.2-3B-Instruct llama3b 4" "google/gemma-3-4b-it gemma4b 4"; do
      set -- $spec
      python run_model.py --model "$1" --tag "$2" --out "$OUT" --npush "$3" 2>&1 | tee -a "$OUT/full.log"
    done
    echo "[all done] $(date) — pod will stop in 60 min" | tee -a "$OUT/full.log"
    touch "$OUT/ALL_DONE"
    sleep 3600 && runpodctl stop pod "$RUNPOD_POD_ID" ;;
esac
