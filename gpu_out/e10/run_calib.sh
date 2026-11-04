#!/bin/zsh
cd "$(dirname "$0")/../../pilot"
PY=${PY:-python}
for spec in qwen1.5b:pilot/e9_qwen1.5b qwen3_4b:gpu_out/qwen3_4b/e9_qwen3_4b llama3b:gpu_out/llama3b/e9_llama3b_8p gemma4b:gpu_out/gemma4b/e9_gemma4b_8p; do
  t=${spec%%:*}; D=${spec#*:}
  $PY e10_calib.py $t ../gpu_out/e10 ../$D ../gpu_out/e10/e10_${t}_p200.json ../gpu_out/e10/e10_calib_$t.json || echo "FAIL $t"
done
echo ALL DONE
