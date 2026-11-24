"""Probe of one model: embedding rank, routing and GPU memory (Transformers, one GPU).

Reads up to `--limit` rows of every task: the prompt as generate.py builds it, then the reference answer, so every
model reads exactly the same tokens. One forward pass per row (bf16, no logits computed) records:

  - pre-logit embeddings (the decoder's final normed hidden state, the input of the LM head) at `--per-row`
    evenly spaced positions per row (prompt and answer, so that a task gives more embeddings than the model has
    dimensions: 200 rows x 64 = 12,800). Their singular values per task give REAM's numerical rank and the
    effective rank (diagnostics.ranks);
  - routing (`--routing`, base models only, since expert ids change after compression): per MoE layer and expert
    the sum of the router's top-k weights, each token's weights normalised to sum to one, per task, language group
    and segment (prompt / answer). With the experts a compression run kept (`retained` of its report) this gives
    the share of the base model's routing that went to the removed experts (diagnostics.removed_share);
  - GPU memory: allocated after loading (the weights) and the peak during the forward passes.

    python probe.py --model base --model-dir ~/models/base --out probes/base --routing \
        --rows my-task=rows.jsonl other-task=other_rows.jsonl

A rows file is the one generate.py reads. The paper's probe read the 200 analysis rows of each task (the rows for
which generate.py keeps top-20 logprobs); `--analysis-rows` selects them the same way.

Outputs `<out>/probe.npz` and `<out>/probe.json` (`complete: true` once done).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

SEGMENTS = ["prompt", "answer"]


def progress(done: int, total: int, what: str):
    print(f"PROGRESS {done}/{total} {what}", flush=True)


def prompt_ids(tokenizer, messages: list[dict]) -> list[int]:  # as generate.prompt_ids
    out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                        tokenize=True, return_dict=True)
    return list(out["input_ids"])


def analysis_rows(row_ids: list[str], n: int) -> set[str]:  # as generate.analysis_rows
    return set(sorted(row_ids, key=lambda r: hashlib.sha256(f"analysis:{r}".encode()).hexdigest())[:n])


def load_rows(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]
    for r in rows:
        r["row_id"] = str(r["row_id"])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="a name for the model, written to probe.json")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows", required=True, nargs="+", metavar="TASK=FILE", help="one rows file per task")
    p.add_argument("--analysis-rows", type=int, default=200,
                   help="per task, read the analysis rows of generate.py (0 = every row of the file)")
    p.add_argument("--limit", type=int, default=None, help="at most N of those rows per task, in row-id order")
    p.add_argument("--per-row", type=int, default=64, help="positions per row kept for the rank")
    p.add_argument("--routing", action="store_true", help="record routing (base models)")
    p.add_argument("--groups", nargs="*", default=["en", "pt", "es", "fr", "id"],
                   help="languages reported one by one in the routing; every other language is 'other'")
    p.add_argument("--device", default="cuda:0", help="the paper's runs used one GPU; 'cpu' for a small test model")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = args.device.startswith("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").unlink(missing_ok=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    weights_bytes = 0
    if cuda:
        torch.cuda.synchronize()
        weights_bytes = torch.cuda.memory_allocated()
    load_s = time.time() - t0
    decoder = model.get_decoder()
    print(f"loaded {args.model} in {load_s:.0f}s, weights {weights_bytes / 2**30:.1f} GiB", flush=True)

    route, layer_names, hooks = {}, [], []
    current = {}  # set per row: task, group, prompt_len
    if args.routing:
        # MoE blocks: a `gate` router next to `experts` (Qwen3MoeSparseMoeBlock, NemotronHMoE); the same module
        # names as the compression report. Both routers return (logits, top-k weights, top-k indices).
        blocks = [(n, m) for n, m in model.named_modules() if hasattr(m, "experts") and hasattr(m, "gate")]
        layer_names = [n for n, _ in blocks]
        n_exp = blocks[0][1].gate.weight.shape[0]
        print(f"routing: {len(blocks)} MoE layers, {n_exp} experts", flush=True)
        for li, (name, block) in enumerate(blocks):
            router = block.gate

            def hook(module, inputs, output, li=li):
                assert isinstance(output, tuple) and len(output) == 3, "expected (logits, top-k weights, indices)"
                w, idx = output[1].detach().float(), output[2].long()
                w, idx = w.reshape(-1, w.shape[-1]), idx.reshape(-1, idx.shape[-1])
                w = w / w.sum(-1, keepdim=True)  # the share of each token's routing; Nemotron scales its weights
                P = current["prompt_len"]
                for seg, sl in (("prompt", slice(0, P)), ("answer", slice(P, None))):
                    acc = torch.zeros(n_exp, device=w.device, dtype=torch.float64)
                    acc.index_add_(0, idx[sl].flatten(), w[sl].flatten().double())
                    key = (current["task"], current["group"], seg)
                    if key not in route:
                        route[key] = torch.zeros(len(layer_names), n_exp, dtype=torch.float64, device=w.device)
                    route[key][li] += acc

            hooks.append(router.register_forward_hook(hook))

    tasks = {}
    for spec in args.rows:
        task, path = spec.split("=", 1)
        rows = load_rows(path)
        if args.analysis_rows:
            keep = analysis_rows([r["row_id"] for r in rows], args.analysis_rows)
            rows = [r for r in rows if r["row_id"] in keep]
        tasks[task] = sorted(rows, key=lambda r: r["row_id"])[:args.limit]

    arrays, info = {}, {"model": args.model, "rows_per_task": {}, "tokens": {}, "longest": 0}
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    total = sum(len(rows) for rows in tasks.values())
    done = 0
    for task, rows in tasks.items():
        embs, n_tok = [], 0
        for r in rows:
            conv = r["messages"]
            p_ids = prompt_ids(tokenizer, conv[:-1])
            a_ids = tokenizer.encode(conv[-1]["content"], add_special_tokens=False)
            ids = torch.tensor([p_ids + a_ids], device=args.device)
            lang = str(r.get("language") or "unknown").lower()
            current.update(task=task, group=lang if lang in args.groups else "other", prompt_len=len(p_ids))
            with torch.inference_mode():
                h = decoder(input_ids=ids).last_hidden_state[0]  # final normed hidden states: the LM head input
            # evenly over the whole sequence: short answers alone would give fewer embeddings than the model has
            # dimensions, which would cap the rank
            pos = np.unique(np.linspace(0, ids.shape[1] - 1, args.per_row).round().astype(int))
            embs.append(h[torch.as_tensor(pos, device=h.device)].float())
            n_tok += ids.shape[1]
            info["longest"] = max(info["longest"], ids.shape[1])
            done += 1
            if done % 10 == 0:
                progress(done, total, "rows")
        e = torch.cat(embs)
        arrays[f"sv__{task}"] = torch.linalg.svdvals(e).cpu().numpy()
        arrays[f"sv_centered__{task}"] = torch.linalg.svdvals(e - e.mean(0, keepdim=True)).cpu().numpy()
        info["rows_per_task"][task] = len(rows)
        info["tokens"][task] = {"read": n_tok, "embeddings": int(e.shape[0]), "dim": int(e.shape[1])}
        print(f"{task}: {len(rows)} rows, {n_tok:,} tokens, {e.shape[0]} embeddings", flush=True)

    for h in hooks:
        h.remove()
    for (task, group, seg), acc in route.items():
        arrays[f"route__{task}__{group}__{seg}"] = acc.cpu().numpy()
    np.savez_compressed(out / "probe.npz", **arrays)
    info.update({"layers": layer_names, "weights_gib": weights_bytes / 2**30,
                 "peak_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0, "load_s": round(load_s, 1),
                 "seconds": round(time.time() - t0, 1), "per_row": args.per_row, "routing": bool(args.routing),
                 "device": torch.cuda.get_device_name(0) if cuda else args.device, "complete": True,
                 "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    (out / "probe.json").write_text(json.dumps(info, indent=1))
    print("DONE " + json.dumps({k: v for k, v in info.items() if k != "layers"}), flush=True)


if __name__ == "__main__":
    main()
