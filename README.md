# prune-dont-merge

Compression code for the paper *Prune, Don't Merge: Expert Compression of Mixture-of-Experts LLMs on Multilingual
Production Tasks* (Tomas Rasymas and Povilas Kvedaras, 2026).

The paper compares two one-shot ways of removing experts from a Mixture-of-Experts (MoE) language model:

- **REAP** ([Lasby et al., arXiv 2510.13999](https://arxiv.org/abs/2510.13999)) deletes the least salient experts.
- **REAM** ([Jha et al., arXiv 2604.04356](https://arxiv.org/abs/2604.04356)) merges them into the most salient ones.

This repository holds what the paper used to compress the models:

- a port of REAM to [llm-compressor](https://github.com/vllm-project/llm-compressor), so that REAM and REAP run on
  the same pipeline and REAM also covers models the original code does not support;
- the script that compresses a model with either method;
- unit checks of the port against the original REAM code.

The evaluation data of the paper are private product logs. They are not here, and neither are the evaluation
scripts, which depend on them.

## Files

| file | what |
|---|---|
| `ream_merge.py` | `REAMMergingModifier`: REAM as an llm-compressor modifier. The module docstring describes the algorithm |
| `reap_modifier.py` | `REAPRecordingModifier`: llm-compressor's REAP, which also writes the experts each layer kept and removed to its report. Pruning itself is unchanged |
| `compress.py` | compresses one model with REAP or REAM on a calibration set of chat conversations |
| `unfuse_experts.py` | rewrites a compressed checkpoint's fused expert tensors into the per-expert layout of the Hub checkpoints, which vLLM loads. `compress.py` calls it |
| `test_ream_merge.py` | unit checks of the port; they run on the CPU |

## Why a port

The [original REAM code](https://github.com/SamsungSAILMontreal/ream) supports Qwen3-style layers only: the MoE block
at `layer.mlp`, gated experts, a softmax router and attention-only decoder layers. The paper also compresses
NVIDIA-Nemotron-3.5-Lightning-30B-A3B, which has Mamba layers, gateless squared-ReLU experts, a sigmoid router with
bias and a shared expert.

The port builds REAM on llm-compressor's REAP code. Both methods then share the calibration pass, the REAP
saliency, the router pruning and the checkpoint format. REAM's group centres are the experts with the highest REAP
saliency, so the two methods differ in what happens to the other experts: REAP deletes them, REAM averages each
into a similar centre.

## Install

Python 3.12 and the versions in `requirements.txt` (llm-compressor 0.15.0, Transformers 5.21.0, PyTorch 2.14.1):

```bash
pip install -r requirements.txt
```

## Compress a model

```bash
python compress.py \
  --model ~/models/Qwen3-30B-A3B-Instruct-2507 \
  --method ream --ratio 0.25 \
  --calibration calibration.jsonl \
  --output ~/models/Qwen3-30B-A3B-Instruct-2507__ream__p25
```

- `--model` is a local directory of the base model, for example from `hf download`.
- `--method` is `reap` or `ream`; `--ratio` is the share of routed experts removed in every MoE layer.
- `--calibration` is a JSON Lines file with one conversation per line:
  `{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.
  Each conversation is formatted with the model's chat template, with thinking off, and is not truncated.
- `--group-size` is REAM's `C`. The default follows the REAM paper: 16 for ratios up to 0.25, otherwise 32.
- `--output` gets the compressed checkpoint. `meta.json` and `report.json` go to `<output>.reports/` (or
  `--reports`). The report lists, per MoE layer, the routing counts, the saliency and the experts kept; for REAM
  also the groups and merge weights, with the expert similarities in `similarity.npz`.

Both methods run on llm-compressor's sequential pipeline: each MoE layer is calibrated on the outputs of the
already compressed layers before it. The router rows of the removed experts are deleted and top-k is unchanged.
Shared experts, attention and Mamba layers are not touched.

The script uses one CUDA GPU. The paper's runs used one 96 GB GPU per 30B model.
Apple's MPS backend is not supported: REAM needs float64.

To use the modifier in your own llm-compressor call, pass it as the recipe, as `compress.py` does:

```python
from llmcompressor import oneshot
from ream_merge import REAMMergingModifier

oneshot(
    model=model,                      # loaded inside llmcompressor.modeling.moe.linearize.load_quantizable_moe()
    processor=tokenizer,
    dataset=calibration,              # tokenized conversations
    recipe=REAMMergingModifier(sparsity=0.25, group_size=16, report_path="report.json"),
    pipeline="sequential",
    moe_calibrate_all_experts=False,  # REAM computes its all-expert statistics in its own hooks
)
```

## Determinism

`compress.py` turns on deterministic kernels (`torch.use_deterministic_algorithms(True)`). Without them, two REAM
runs of the same configuration agreed in only 18 of the 48 MoE layers of Qwen3-30B-A3B-Instruct-2507: REAM's greedy
grouping amplifies last-bit differences from layer to layer. With them, two runs are bit-identical, at about 25%
extra time.

## Differences from the original REAM code

All are deliberate; the module docstring of `ream_merge.py` gives the details.

1. **Router scores.** Dense router scores use the model's own scoring function: softmax for Qwen routers, sigmoid
   for routers with an `e_score_correction_bias` (Nemotron, GLM). The original always applies softmax.
2. **Token sampling.** The neuron activations used for alignment are sampled on the same tokens for all experts of a
   layer. The original samples tokens separately for each expert.
3. **Saliency.** REAP saliency comes from the router's actual top-k weights, as in llm-compressor's REAP, so REAM's
   centres are the experts REAP would keep at that layer.
4. **Averaging precision.** The weighted average is accumulated in float32 and cast once; the original casts after
   every addition.
5. **Not supported:** group-limited routing (`n_group > 1`) and multi-GPU calibration.

The group size is not a difference: as in the original code, a group holds at most `C` experts *including* its
centre. The REAM paper's text reads "up to C unassigned non-centroid experts".

## Checks

```bash
git clone https://github.com/SamsungSAILMontreal/ream && git -C ream checkout 84a3030
python test_ream_merge.py --ream-repo ream
```

1. **Grouping parity:** the port's grouping gives the same groups as the original `pseudo_group` on the same
   statistics.
2. **PCA parity** with the original `pca_reduce`.
3. **Neuron alignment:** an expert whose neurons are a shuffled copy of its centre is matched back exactly, and
   merging it returns the centre, for gated and for squared-ReLU experts.
4. **Statistics and sampling:** the batched all-expert pass equals running every expert on every token, and all
   experts of a layer keep the activations of the same tokens.

Without `--ream-repo`, checks 1 and 2 are skipped.

What was run:

- The paper's eight compressed checkpoints (Qwen3-30B-A3B-Instruct-2507 and NVIDIA-Nemotron-3.5-Lightning-30B-A3B,
  REAP and REAM, 25% and 50%) were made with these modifiers. All load without missing or unexpected weights.
- `compress.py` here differs from the script of those runs in how it reads the calibration set (a JSON Lines file
  instead of the private data), in its command-line paths, and in accepting single-file checkpoints. This version
  was run end to end on a tiny random Qwen3-MoE model for both methods, on the CPU. It was not rerun on the 30B
  models.

## Licence

MIT, see `LICENSE`. The grouping, PCA and neuron-alignment code in `ream_merge.py` is adapted from
[SamsungSAILMontreal/ream](https://github.com/SamsungSAILMontreal/ream) at commit `84a3030` (MIT, Copyright (c) 2026
Samsung Electronics Co., Ltd.).

## Citation

```bibtex
@article{rasymas2026prune,
  title  = {Prune, Don't Merge: Expert Compression of Mixture-of-Experts LLMs on Multilingual Production Tasks},
  author = {Rasymas, Tomas and Kvedaras, Povilas},
  year   = {2026}
}
```

Please also cite REAP and REAM, whose methods this code implements.
