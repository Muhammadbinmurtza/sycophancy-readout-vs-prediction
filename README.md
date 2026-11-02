# Reading the Reversal, Not Predicting It

Code, labels and results for *"Reading the Reversal, Not Predicting It: Sycophancy Probes Trained on Simulated Dialogues Are an Unreliable Route to a Predictive Signal"* (Muhammad Bin Murtaza, 2026).

We audit head-level sycophancy probes (Genadi et al., EACL 2026) on four instruction-tuned models: Qwen2.5-1.5B, Qwen3-4B, Llama-3.2-3B and Gemma-3-4B. We separate whether a probe *reads* a correct→incorrect reversal already written in the transcript from whether it *predicts* that a model is about to capitulate under user pushback.

## Main results

- **Read-out:** probes trained on teacher-forced simulated dialogues reach AUROC 0.94–0.98 in distribution. About 0.75 is already available at layer 0, and the probes respond mostly to a *changed* answer. On real episodes they read reversals at the answer position (0.67–0.93).
- **No prediction from a single greedy outcome:** the same probes are at or below chance before the reply (0.38–0.57). A matched within-question test (E9) is dominated by pushback wording.
- **A predictive signal exists (E10):** with 10 samples per (question, pushback) cell and question and wording effects removed, an in-domain probe on the pre-reply residual stream predicts P(cave) on all four models (r = 0.20–0.33, p = 0.005). Simulated-dialogue probes recover none of it on Qwen and part of it on Llama and Gemma.

## Layout

| Path | Contents |
|---|---|
| `pilot/` | Analysis code (Mac/CPU) plus Qwen2.5-1.5B data and results |
| `gpu/` | Generation, feature extraction, sampling and judging for the GPU models (RunPod, RTX 4090) |
| `gpu_out/<model>/` | Real episodes with labels, E9 episodes, and result JSONs for Qwen3-4B, Llama-3.2-3B and Gemma-3-4B |
| `gpu_out/judged/` | LLM-judge labels (Qwen2.5-14B-Instruct-AWQ, prompt variants A/B) and judge-vs-Claude agreement |
| `gpu_out/e10/` | E10 sampled replies, judge labels and results |
| `handlabels/` | Claude labels in batches, the annotation helpers, and the blind human check (`human/`) |
| `figs/make_figs.py` | Paper figures |

## Experiments → code → outputs

| Exp. | What | Code | Main output |
|---|---|---|---|
| E1–E5 | Simulated dialogues, layer-0 baseline, truth transfer, head scan | `pilot/genadi_confound.py`, `pilot/analyze_confound.py`, `gpu/run_model.py` (stage `sim`) | `results_<model>.json` |
| Factorial | CI/CC/IC/II decomposition of the head score | `pilot/factorial_confound.py` | `factorial_<model>.json` |
| E8 | Simulated→real transfer vs. in-domain, answer and pre-reply | `pilot/generate_real_tqa.py`, `gpu/run_model.py` (stage `real`), `pilot/e8_real_tqa.py` | `e8_<model>_claude.json` |
| E9 | Matched within-question pre-reply test | `pilot/e9_generate.py`, `gpu/run_model.py` (stage `e9`), `pilot/e9_analyse.py` | `e9_<model>_elig*.json` |
| E10 | P(cave) beyond question and wording effects | `gpu/e10.py`, `pilot/e10_analyse.py`, `pilot/e10_genadi.py`, `pilot/e10_lexical.py` | `gpu_out/e10/e10_<model>_p200.json` |
| Labels | Judge, agreement, human check | `gpu/judge.py`, `pilot/judge_agree.py`, `handlabels/human/` | `gpu_out/judged/` |

## Labels

- Every real turn-1 and turn-2 answer (4 models × 817 questions × 2 turns) was labelled CORRECT, INCORRECT or UNCLEAR against the TruthfulQA references by Claude (Anthropic). Fields: `first_verdict`, `final_verdict` in `*.claude.jsonl`.
- These labels were **not** blind to the hypothesis. A blind human check of 50 answers (`handlabels/human/`) agrees with them on 29 of 33 committed items (κ = 0.76).
- E9 finals for the GPU models and all E10 samples were graded by an LLM judge (`gpu_out/judged/`, `gpu_out/e10/*_judged.jsonl`).

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **Analyses from the released episodes and labels** (CPU): run the scripts in `pilot/`. Each file's docstring shows its exact command line.
- **Activations are not included** (several GB per model), so feature-based analyses first need the features regenerated:
  - Qwen2.5-1.5B: `pilot/genadi_confound.py`, `pilot/generate_real_tqa.py` / `pilot/e8_real_tqa.py extract`, `pilot/e9_generate.py`. Apple M1 Pro, fp32.
  - Other models: `gpu/run_all.sh full` on one 24 GB GPU (HF token required for Llama and Gemma), then `gpu/run_judge.sh`, `gpu/run_e10.sh` and `gpu/run_e10b.sh`.
- **Figures:** `cd figs && python make_figs.py`.

Total GPU cost for every experiment in the paper was about USD 6 on a rented RTX 4090.

### Numerical notes
- Activations are cached in fp16. Gemma-3-4B residual layers 27–33 overflow; the scripts clip ±inf and the paper does not interpret those layers.
- E10 samples were drawn in bf16 with vLLM 0.11, which needs transformers < 4.58.
- Model-generated text is released as generated, except that two e-mail addresses a model invented were replaced with `[email]`.

## Data and licences

- **Code:** MIT licence (`LICENSE`).
- **Labels and results:** CC BY 4.0.
- **Questions and reference answers:** from TruthfulQA (Lin et al., 2022; Apache 2.0).
- **Model outputs:** generated by Qwen, Llama 3.2 and Gemma 3 under their respective licences.

## Citation

```bibtex
@article{binmurtaza2026reading,
  title  = {Reading the Reversal, Not Predicting It: Sycophancy Probes Trained on Simulated Dialogues Are an Unreliable Route to a Predictive Signal},
  author = {Bin Murtaza, Muhammad},
  journal= {arXiv preprint},
  year   = {2026}
}
```
