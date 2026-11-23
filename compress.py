"""Compress one MoE model with REAP (pruning) or REAM (merging) on a calibration set of chat conversations.

Uses one CUDA GPU (pick it with CUDA_VISIBLE_DEVICES); without CUDA it runs on the CPU, which is only practical
for tiny test models:
    python compress.py --model ~/models/Qwen3-30B-A3B-Instruct-2507 --method ream --ratio 0.25 \
        --calibration calibration.jsonl --output ~/models/Qwen3-30B-A3B-Instruct-2507__ream__p25

--model is a local directory of the base model (for example from `hf download`).
--calibration is a JSON Lines file with one conversation per line: {"messages": [{"role": ..., "content": ...}, ...]}.
Each conversation is formatted with the model's chat template (thinking off) and is not truncated, so a sample is
the whole conversation: system prompt, user message and assistant answer.

Both methods run on llm-compressor's sequential pipeline: each MoE layer is calibrated on the outputs of the
already compressed layers before it. Both remove the same share of the routed experts in every MoE layer and cut
the router to match.

Writes the compressed checkpoint to --output and, to --reports (default: <output>.reports/):
    meta.json    experts, parameters, size, time, peak GPU memory, calibration size
    report.json  per MoE layer: routing counts, saliency, the experts kept and removed; for REAM also the groups
                 and merge weights (and similarity.npz with the expert similarities the grouping used)
Runs are deterministic (deterministic CUDA kernels), so a rerun gives the same experts and groups.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path


def calibration_inputs(tokenizer, path: Path, limit_rows: int | None = None) -> list[list[int]]:
    """Token ids of the calibration conversations: the model's chat template with thinking off, not truncated."""
    input_ids = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            input_ids.append(list(tokenizer.apply_chat_template(
                messages, tokenize=True, enable_thinking=False, return_dict=True)["input_ids"]))
            if limit_rows and len(input_ids) >= limit_rows:
                break
    if not input_ids:
        raise ValueError(f"no conversations in {path}")
    return input_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path, help="local directory of the base model")
    p.add_argument("--method", required=True, choices=["reap", "ream"])
    p.add_argument("--ratio", required=True, type=float, help="share of the routed experts removed per MoE layer")
    p.add_argument("--calibration", required=True, type=Path, help='JSON Lines, one {"messages": [...]} per line')
    p.add_argument("--output", required=True, type=Path, help="directory of the compressed checkpoint")
    p.add_argument("--reports", type=Path, default=None, help="directory of meta.json and report.json")
    p.add_argument("--group-size", type=int, default=None, help="REAM's C; default 16 at ratio <= 0.25, else 32")
    p.add_argument("--limit-rows", type=int, default=None, help="first N calibration conversations (quick tests)")
    p.add_argument("--overwrite", action="store_true", help="replace --output if it exists")
    args = p.parse_args()

    # deterministic CUDA kernels before CUDA starts: REAM's greedy grouping amplifies run-to-run noise
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modeling import repack_moe
    from llmcompressor.modeling.moe.linearize import load_quantizable_moe
    from llmcompressor.modifiers.pruning.reap.utils import get_moe_attrs
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ream_merge import REAMMergingModifier
    from reap_modifier import REAPRecordingModifier
    from unfuse_experts import unfuse_checkpoint

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0)

    src = args.model.expanduser()
    out = args.output.expanduser()
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it")
    tmp = out.with_name(out.name + ".tmp")
    reports = (args.reports or out.with_name(out.name + ".reports")).expanduser()
    shutil.rmtree(tmp, ignore_errors=True)
    reports.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(src)
    input_ids = calibration_inputs(tokenizer, args.calibration.expanduser(), args.limit_rows)
    calib = Dataset.from_dict({"input_ids": input_ids, "attention_mask": [[1] * len(x) for x in input_ids]})
    n_tokens = sum(map(len, input_ids))
    max_len = max(map(len, input_ids))
    print(f"calibration: {len(calib)} samples, {n_tokens:,} tokens, longest {max_len:,}", flush=True)

    t0 = time.time()
    with load_quantizable_moe():
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    load_s = time.time() - t0
    attrs = get_moe_attrs(model, [])
    experts_before = attrs.num_experts
    params_before = sum(p.numel() for p in model.parameters())

    if args.method == "ream":
        group_size = args.group_size or (16 if args.ratio <= 0.25 else 32)
        recipe = REAMMergingModifier(sparsity=args.ratio, group_size=group_size, report_path=str(reports / "report.json"))
    else:
        group_size = None
        recipe = REAPRecordingModifier(sparsity=args.ratio, report_path=str(reports / "report.json"))

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    oneshot(
        model=model,
        processor=tokenizer,  # the calibration set is tokenized already
        dataset=calib,
        recipe=recipe,
        num_calibration_samples=len(calib),
        max_seq_length=max_len,
        shuffle_calibration_samples=False,
        pipeline="sequential",
        moe_calibrate_all_experts=False,  # REAM computes its all-expert statistics in its own hooks
        sequential_offload_device="cuda:0" if cuda else "cpu",
    )
    compress_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() if cuda else None

    repack_moe(model)
    for module in model.modules():  # repack_moe adds an empty MoNe `novices` parameter
        if getattr(module, "novices", None) is not None and module.novices.numel() == 0:
            del module.novices
    if model.config.get_text_config().tie_word_embeddings:
        model.tie_weights()
    cfg = model.config.get_text_config()
    experts_after = next(getattr(cfg, k) for k in ("num_experts", "n_routed_experts", "num_local_experts") if hasattr(cfg, k))
    params_after = sum(p.numel() for p in model.parameters())
    model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    # the per-expert layout of the original Hub checkpoints, which vLLM loads
    unfuse_checkpoint(tmp)
    # keep the base model's default sampling settings
    if (src / "generation_config.json").exists():
        shutil.copy(src / "generation_config.json", tmp / "generation_config.json")

    def size(path):
        return sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))

    meta = {
        "model": src.name, "method": args.method, "ratio": args.ratio, "group_size": group_size,
        "experts_before": experts_before, "experts_after": experts_after, "moe_layers": len(attrs.moe_layer_names),
        "params_before": params_before, "params_after": params_after,
        "size_bytes_before": size(src), "size_bytes_after": size(tmp),
        "load_s": round(load_s), "compress_s": round(compress_s), "peak_gpu_bytes": peak,
        "calibration": {"samples": len(calib), "tokens": n_tokens, "longest": max_len, "limit_rows": args.limit_rows},
        "deterministic": True, "expert_layout": "per-expert (as the original checkpoint)",
    }
    (reports / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)  # only a complete checkpoint gets the final name
    print("DONE " + json.dumps({k: meta[k] for k in ("model", "method", "ratio", "experts_after", "compress_s")}),
          flush=True)


if __name__ == "__main__":
    main()
