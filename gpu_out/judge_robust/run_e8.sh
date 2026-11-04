#!/bin/zsh
# E8 re-run with judge labels instead of Claude labels (robustness to the labeller).
cd "$(dirname "$0")/../../pilot"
PY=${PY:-python}
R=../gpu_out/judge_robust
for m in qwen1.5b qwen3_4b llama3b gemma4b; do
  if [ $m = qwen1.5b ]; then d=.; else d=../gpu_out/$m; fi
  for v in cons A; do
    echo "== $m $v $(date +%T)"
    $PY e8_real_tqa.py analyse $d/cache_$m.pt $d/results_$m.json $R/real_tqa_$m.judge$v.jsonl $d/e8_feats_$m.npz $R/e8_${m}_judge$v.json > $R/e8_${m}_judge$v.log 2>&1 || echo FAIL
  done
done
echo ALL DONE
