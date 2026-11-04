"""Agreement of the blind human labels (items 1-300) with Claude and with the LLM judge.
Usage: python handlabels/human/agree300.py   (run from the sycophancy-locus root)"""
import json
from collections import Counter

H = 'handlabels/human/'
items = {x['id']: x for x in json.load(open(H + 'items_300.json'))}
human = {x['id']: x['human'] for f in ('human_labels.json', 'human_labels_new250.json') for x in json.load(open(H + f))}
for i, x in items.items():
    assert i in human, i
jfile = {'qwen1.5b': 'calib_real', 'qwen3_4b': 'qwen3_4b_real', 'llama3b': 'llama3b_real', 'gemma4b': 'gemma4b_real'}
judge = {}
for m, t in jfile.items():
    for l in open(f'gpu_out/judged/judged_{t}.jsonl'):
        r = json.loads(l)
        judge[(m, r['qidx'], r['turn'])] = r['A'] if r['A'] == r['B'] else 'UNCLEAR'   # A/B consensus

def kappa(a, b):
    n, cats = len(a), set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum(a.count(c) / n * b.count(c) / n for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)

def compare(a, b):
    both = [i for i in range(len(a)) if a[i] != 'UNCLEAR' and b[i] != 'UNCLEAR']
    ca, cb = [a[i] for i in both], [b[i] for i in both]
    return {'n': len(a), 'agree3': sum(x == y for x, y in zip(a, b)), 'kappa3': round(kappa(a, b), 3),
            'n_committed': len(both), 'agree_committed': sum(x == y for x, y in zip(ca, cb)),
            'kappa_committed': round(kappa(ca, cb), 3) if both else None}

out = {}
for name, ids in [('first50', range(1, 51)), ('new250', range(51, 301)), ('all300', range(1, 301))]:
    ids = list(ids)
    h = [human[i] for i in ids]; c = [items[i]['claude'] for i in ids]
    j = [judge[(items[i]['model'], items[i]['qidx'], items[i]['turn'])] for i in ids]
    out[name] = {'human_vs_claude': compare(h, c), 'human_vs_judge': compare(h, j), 'claude_vs_judge': compare(c, j),
                 'human_counts': Counter(h), 'claude_counts': Counter(c)}
    flips = [i for i in ids if {human[i], items[i]['claude']} == {'CORRECT', 'INCORRECT'}]
    out[name]['human_claude_flips'] = flips
per = {}
for m in jfile:
    ids = [i for i in range(1, 301) if items[i]['model'] == m]
    per[m] = compare([human[i] for i in ids], [items[i]['claude'] for i in ids])
out['all300_by_model_human_vs_claude'] = per
json.dump(out, open(H + 'agree300.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
