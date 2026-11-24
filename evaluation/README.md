# Evaluation code

The parts of the paper's evaluation that do not depend on the products it was run on: generation, the statistics,
the diagnostics, and the checks that need no LLM. With your own rows and your own scores they repeat the paper's
evaluation on other data.

## What is here, and what is not

| file | what |
|---|---|
| `generate.py` | generation with vLLM: greedy or the model's default sampling with a per-row seed, an optional JSON schema, the token log-probabilities; and the base model scoring a compressed model's answers (cross-perplexity, logit divergence) |
| `probe.py` | one forward pass per row with Transformers: pre-logit embeddings for the rank, routing per expert, GPU memory of the weights |
| `stats.py` | paired bootstrap intervals, McNemar's exact test, the sign-flip permutation test, Holm's correction |
| `diagnostics.py` | repetition loops, distinct-n, perplexity and cross-perplexity, completion-logit JSD, numerical and effective rank, the routing share of removed experts |
| `checks.py` | exact-match scoring of the three discriminative tasks, the rule checks and chrF of domain descriptions, the two relaxed gates of the email-template score |
| `test_evaluation.py` | checks of all of the above on made-up data, on the CPU |
| `example/` | made-up rows, a JSON schema and made-up scores, in the formats the scripts read |

Not here:

- **The email-template rubric** (13 categories with static checks, checks on the rendered email, and LLM text and
  vision judges) and **the LLM judge's instructions and the products' criteria**. They are production code and
  configuration of the products. The paper describes the rubric and its gates in Appendix D.
- **The data**: the prompts, the references and the per-row results. They are private product logs with customer
  input.

So the scores of the two judged tasks and of the email templates cannot be produced with this folder alone. What
it gives for them: how the answers were generated, how the scores were compared, the two gate rules the paper
relaxed (`checks.py`), and the judge-free metrics of the paper's Appendix J for categorisation and domain
descriptions.

## Install

Python 3.12. `stats.py`, `diagnostics.py` and `checks.py` need the first group only.

```bash
pip install numpy scipy sacrebleu beautifulsoup4      # statistics, diagnostics, checks
pip install vllm==0.32.0                              # generate.py (the paper's version)
pip install torch==2.14.1 transformers==5.21.0        # probe.py (the paper's versions)
```

`checks.lingua_detect` also needs `lingua-language-detector`.

## Rows

`generate.py` and `probe.py` read a task from a JSON Lines file, one row per line (`example/rows.jsonl`):

```json
{"row_id": "row-000", "language": "en", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "<reference answer>"}]}
```

The prompt is every message but the last. The last message is the reference answer; it is never shown to the
model during generation.

## Generate

```bash
python generate.py generate --model base   --model-dir ~/models/base   --task my-task --rows rows.jsonl --decoding greedy
python generate.py generate --model pruned --model-dir ~/models/pruned --task my-task --rows rows.jsonl --decoding greedy
python generate.py cross    --model base   --model-dir ~/models/base   --task my-task --rows rows.jsonl --targets pruned
```

- `--decoding greedy` for tasks with one right answer; `--decoding model_default` uses the temperature, top-p and
  top-k of the model's `generation_config.json`. The seed of a row comes from its row id, so every model draws the
  same random numbers on a row. Thinking is off.
- `--schema schema.json` constrains the answer to a JSON schema (vLLM structured outputs) and caps it at 1,024
  tokens. Without a schema the cap is 32,768 tokens (`--max-tokens`).
- `cross` loads the base model and scores the answers of the `--targets` models, token by token.
- Output goes to `results/<model>/<task>/`; the module docstring lists the files. `generation.json` has the
  generation time and the throughput.

## Probe

```bash
python probe.py --model base --model-dir ~/models/base --out probes/base --routing --rows my-task=rows.jsonl
```

Per task it reads the 200 analysis rows (a fixed pseudo-random subset, the same as in `generate.py`), the prompt
and then the reference answer. `--routing` is for base models: expert ids change after compression.

## Statistics

Every comparison is paired, on the same rows. `stats.compare(x, y, binary)` returns the difference of the means
with a 95% paired bootstrap interval (10,000 resamples), the retention with its interval, and the p-value:
McNemar's exact test for pass/fail metrics, a paired sign-flip permutation test on the mean for scores.
`stats.holm` corrects one family of comparisons.

```bash
python stats.py --scores example/scores.jsonl --binary --pairs base:pruned base:merged merged:pruned
```

```
pair (x:y)                rows       x       y            y - x [95% CI]        retention % [95% CI]         p    Holm p
base:pruned                200    84.0    78.5        -5.5 [-10.5, -1.0]           93.5 [87.7, 98.9]     0.043     0.043
base:merged                200    84.0    66.5      -17.5 [-24.0, -11.5]           79.2 [71.9, 86.4]   1.8e-07   5.3e-07
merged:pruned              200    66.5    78.5       +12.0 [+7.0, +17.5]        118.0 [109.7, 128.2]   1.9e-05   3.9e-05
test: McNemar exact; Holm correction over these 3 comparisons
```

The paper has two families: the 40 comparisons of a compressed model with its base model (5 tasks, 8 compressed
models) and the 20 comparisons of REAP with REAM (5 tasks, 2 base models, 2 ratios). Each is corrected on its own.

## Diagnostics

```python
import json
import diagnostics as d

rows = [json.loads(line) for line in open("results/pruned/my-task/generations.jsonl")]
kept = [r["row_id"] for r in rows if r["finish_reason"] != "length"]          # answers under the token cap
own = d.perplexity("results/pruned/my-task/tokens.npz", kept)                 # the model's own perplexity
cross = d.perplexity("results/pruned/my-task/cross.npz", kept)                # cross-perplexity (the base model's)
jsd = d.mean_jsd("results/pruned/my-task/top.npz", "results/pruned/my-task/cross.npz")
loops = [d.is_loop(r["text"], r["finish_reason"] == "length") for r in rows]
distinct2 = d.distinct_n([r["text"] for r, loop in zip(rows, loops) if not loop], 2)
```

For a task decoded under a schema, pass `skip=1` to the perplexity and JSD functions (module docstring).

## Checks

```bash
python test_evaluation.py
```

1. **Statistics** on arrays with known answers.
2. **Checks** on small examples: the exact-match scores, every rule of domain descriptions, the two relaxed gates.
3. **Diagnostics** on arrays with known answers.
4. **`generate.py`**, both modes, with a stand-in for vLLM: a "model" whose next-token probabilities are a known
   function, so the perplexities and the JSD it must give are known. This checks how the script reads its rows and
   writes its files; it does not check vLLM.
5. **`probe.py`** on a tiny random Qwen3-MoE model on the CPU. Skipped without PyTorch and Transformers.

What was run:

- The answers, statistics and diagnostics of the paper were produced with the code these files are taken from.
- `stats.py`, `diagnostics.py` and `checks.py` were run again, as they are here, on the paper's stored per-row
  results. They reproduce the paper's numbers: the 40 comparisons with the base models and the 20 head-to-head
  comparisons (differences, intervals, retentions, Holm-corrected verdicts), the loop flag of all 100,000 answers,
  distinct-2, the ranks, the routing shares, the exact-match scores, the rule checks of domain descriptions, and
  the relaxed `html_complete` gate on the 9,423 answers that fail the production gate. Perplexity and JSD were
  checked on made-up arrays only (check 3), because the token files of the generation runs were not at hand, and
  the relaxed unsubscribe gate on small examples only (check 2). `checks.chrf` is sacrebleu's chrF with its
  defaults.
- `generate.py` and `probe.py` differ from the scripts of the paper's runs in how they read their rows (a JSON
  Lines file instead of the private data) and in their command-line options. These versions were run with checks 4
  and 5 above. They were not rerun on the 30B models.
- `checks.lingua_detect` is a stand-in for the language detector of the paper's runs, which belongs to the rubric
  library and is not released. The stored language verdicts of the paper's runs were used in the reproduction
  above; the stand-in was not compared with them.
