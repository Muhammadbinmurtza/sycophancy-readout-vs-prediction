"""Write real_tqa_<model>.judge{cons,A}.jsonl: the real episodes with first/final verdicts taken from the
LLM judge (A/B consensus, or prompt A alone) instead of Claude. Run from the repository root."""
import json
jf = {'qwen1.5b': ('pilot/real_tqa_qwen1.5b.claude.jsonl', 'calib_real'),
      'qwen3_4b': ('gpu_out/qwen3_4b/real_tqa_qwen3_4b.claude.jsonl', 'qwen3_4b_real'),
      'llama3b': ('gpu_out/llama3b/real_tqa_llama3b.claude.jsonl', 'llama3b_real'),
      'gemma4b': ('gpu_out/gemma4b/real_tqa_gemma4b.claude.jsonl', 'gemma4b_real')}
for m, (src, t) in jf.items():
    J = {}
    for l in open(f'gpu_out/judged/judged_{t}.jsonl'):
        r = json.loads(l); J[(r['qidx'], r['turn'])] = r
    for v in ('cons', 'A'):
        with open(f'gpu_out/judge_robust/real_tqa_{m}.judge{v}.jsonl', 'w') as out:
            for l in open(src):
                r = json.loads(l)
                for turn in ('first', 'final'):
                    j = J[(r['qidx'], turn)]
                    lab = (j['A'] if j['A'] == j['B'] else 'UNCLEAR') if v == 'cons' else j['A']
                    r[f'{turn}_verdict'] = lab if lab in ('CORRECT', 'INCORRECT') else 'UNCLEAR'
                out.write(json.dumps(r) + '\n')
