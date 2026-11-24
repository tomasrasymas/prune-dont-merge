"""Generation for the evaluation, with vLLM: one model, one task, one GPU.

Two modes.

  generate   the model answers every row of the task. Decoding is greedy (`--decoding greedy`) or the model's
             default sampling from its generation_config.json (`--decoding model_default`), with a per-row seed
             derived from the row id, so that every model draws the same random numbers on a row. Thinking is
             off; answers are capped at `--max-tokens` (32,768), or at 1,024 under a JSON schema (`--schema`).
             Also kept:
               - the logprob of every generated token (the model's own view of its answer);
               - the top-20 logprobs at the first 2,048 answer positions, on a fixed pseudo-random subset of 200
                 rows, the analysis rows (the completion-logit JSD of the REAP paper);
               - the NLL of the reference answer under the model (teacher forced).
  cross      the base model scores the answers of its compressed models (teacher forced on their token ids): the
             logprob of every answer token (cross-perplexity, REAP) and, on the analysis rows, its logprobs of the
             compressed model's top-20 tokens (JSD over the same support).

    python generate.py generate --model base --model-dir ~/models/base --task my-task --rows rows.jsonl \
        --decoding greedy --results results
    python generate.py generate --model pruned --model-dir ~/models/pruned --task my-task --rows rows.jsonl \
        --decoding greedy --results results
    python generate.py cross --model base --model-dir ~/models/base --task my-task --rows rows.jsonl \
        --results results --targets pruned

`--rows` is a JSON Lines file with one row per line:
  {"row_id": "...", "language": "en", "messages": [{"role": "system", "content": "..."}, {"role": "user", ...},
   {"role": "assistant", "content": "<reference answer>"}]}
The prompt is every message but the last; the last message is the reference answer.

Output, in <results>/<model>/<task>/:
  generations.jsonl   one record per row: row_id, language, prompt and answer tokens, text, finish reason, sum of
                      the answer's own logprobs, reference tokens and reference logprob sum
  tokens.npz          answer token ids and their own logprobs, all rows (concatenated, `offsets` delimit rows)
  top.npz             analysis rows: top-20 token ids and logprobs per answer position
  generation.json     run info: model, decoding, vLLM version, times, throughput; `complete` once done
  cross.npz           (cross mode, written next to each target's files) the base model's logprob of every answer
                      token, and its logprobs of the top-20 ids on the analysis rows
  cross.json          cross-mode run info
Generation runs in chunks of 500 rows saved under parts/, so a stopped run resumes at the next chunk.
diagnostics.py reads these files.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

TOP_K = 20
TOP_POSITIONS = 2048
CHUNK_ROWS = 500
MAX_MODEL_LEN = 65536  # the longest prompt of the paper (about 25k tokens) plus the 32k answer cap
MAX_NEW_TOKENS = 32768
SCHEMA_MAX_TOKENS = 1024
ANALYSIS_ROWS = 200  # rows per task that also keep top-k logprobs, for the completion-logit JSD


def progress(done: int, total: int, what: str):
    print(f"PROGRESS {done}/{total} {what}", flush=True)


def load_rows(path: str, limit_rows: int | None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]
    for r in rows:
        r["row_id"] = str(r["row_id"])
        assert r["messages"][-1]["role"] == "assistant", "the last message of a row is the reference answer"
    assert len({r["row_id"] for r in rows}) == len(rows), "row ids must be unique"
    return rows[:limit_rows] if limit_rows else rows


def sampling_settings(decoding: str, model_dir: Path) -> dict:
    """temperature / top_p / top_k (greedy = temperature 0; model_default = the model's generation_config.json)."""
    if decoding == "greedy":
        return {"temperature": 0.0}
    gen = json.loads((Path(model_dir) / "generation_config.json").read_text())
    settings = {"temperature": gen.get("temperature", 1.0)}
    for key in ("top_p", "top_k", "min_p"):
        if gen.get(key) is not None:
            settings[key] = gen[key]
    return settings


def row_seed(row_id: str) -> int:
    """Stable per-row sampling seed, the same for every model."""
    return int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16)


def analysis_rows(row_ids: list[str], n: int = ANALYSIS_ROWS) -> set[str]:
    """A fixed pseudo-random subset of `n` rows (lowest hash of the row id), the same for every model."""
    return set(sorted(row_ids, key=lambda r: hashlib.sha256(f"analysis:{r}".encode()).hexdigest())[:n])


def make_llm(model_dir: Path, gpu_memory_utilization: float):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # FlashInfer's sampler would need ninja to compile
    from vllm import LLM

    cfg = json.loads((model_dir / "config.json").read_text())
    max_len = min(MAX_MODEL_LEN, cfg.get("text_config", cfg)["max_position_embeddings"])
    llm = LLM(model=str(model_dir), max_model_len=max_len, gpu_memory_utilization=gpu_memory_utilization, seed=0,
              max_logprobs=TOP_K + 1)  # cross mode scores the answer token plus TOP_K ids per position
    return llm, max_len


def prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                        tokenize=True, return_dict=True)
    return list(out["input_ids"])


def entries(flat, position: int) -> dict[int, float]:
    """token id -> logprob at one position of a FlatLogprobs."""
    s, e = flat.start_indices[position], flat.end_indices[position]
    return dict(zip(flat.token_ids[s:e], flat.logprobs[s:e]))


def concat(arrays: list[np.ndarray], dtype) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(a) for a in arrays])
    flat = np.concatenate(arrays).astype(dtype) if arrays else np.zeros(0, dtype)
    return flat, offsets


# ---------------------------------------------------------------------------------------------------------- generate

def generate(args):
    from vllm import SamplingParams, __version__ as vllm_version
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import StructuredOutputsParams

    model_dir = Path(args.model_dir).expanduser()
    out = Path(args.out_dir)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.rows, args.limit_rows)
    row_ids = [r["row_id"] for r in rows]
    subset = analysis_rows(row_ids)
    settings = sampling_settings(args.decoding, model_dir)
    max_tokens = args.max_tokens or MAX_NEW_TOKENS
    schema = json.loads(Path(args.schema).read_text()) if args.schema else None  # structured output
    if schema:
        # JSON whitespace stays allowed: forcing compact JSON bans the model's own `": "` tokens and derails it.
        # A degraded model could instead fill that whitespace forever, so the answer is capped well above any real
        # one but far below 32k
        max_tokens = min(max_tokens, SCHEMA_MAX_TOKENS)
    n_chunks = (len(rows) + CHUNK_ROWS - 1) // CHUNK_ROWS
    # chunks of an earlier, stopped run are reused only if it ran with the same settings
    fingerprint = {"model_dir": str(model_dir), "rows": row_ids, "decoding": settings, "max_tokens": max_tokens,
                   "response_schema": schema, "top_k": TOP_K, "top_positions": TOP_POSITIONS, "chunk_rows": CHUNK_ROWS}
    fp_file = parts / "settings.json"
    if fp_file.exists() and json.loads(fp_file.read_text()) != fingerprint:
        print("discarding chunks of an earlier run with other settings", flush=True)
        shutil.rmtree(parts)
        parts.mkdir(parents=True)
    fp_file.write_text(json.dumps(fingerprint))

    t0 = time.time()
    llm, max_len = make_llm(model_dir, args.gpu_memory_utilization)
    load_s = time.time() - t0
    tokenizer = llm.get_tokenizer()
    print(f"loaded {model_dir.name} in {load_s:.0f}s; decoding {settings}, schema {bool(schema)}, max_tokens {max_tokens}, "
          f"max_model_len {max_len}; {len(rows)} rows in {n_chunks} chunks", flush=True)

    gen_s = 0.0
    for k in range(n_chunks):
        part = parts / f"gen_{k:03d}.npz"
        if part.exists():
            gen_s += float(np.load(part)["seconds"])
            progress(min((k + 1) * CHUNK_ROWS, len(rows)), len(rows), "rows generated (resumed)")
            continue
        chunk = rows[k * CHUNK_ROWS:(k + 1) * CHUNK_ROWS]
        prompts, params = [], []
        for r in chunk:
            rid = r["row_id"]
            ids = prompt_ids(tokenizer, r["messages"][:-1])
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(SamplingParams(**settings, max_tokens=max(1, min(max_tokens, max_len - len(ids))),
                                         seed=row_seed(rid), logprobs=TOP_K if rid in subset else 0,
                                         flat_logprobs=True,
                                         structured_outputs=StructuredOutputsParams(json=schema) if schema else None))
        t1 = time.time()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        seconds = time.time() - t1
        gen_s += seconds

        records, own_lp, top_rows, top_ids, top_lp = [], [], [], [], []
        for r, prompt, o in zip(chunk, prompts, outputs):
            rid = r["row_id"]
            c = o.outputs[0]
            ids = list(c.token_ids)
            lp = np.array([entries(c.logprobs, p).get(t, np.nan) for p, t in enumerate(ids)], dtype=np.float32)
            own_lp.append((np.array(ids, dtype=np.int32), lp))
            if rid in subset:
                n = min(len(ids), TOP_POSITIONS)
                ti = np.full((n, TOP_K), -1, dtype=np.int32)
                tl = np.full((n, TOP_K), np.nan, dtype=np.float32)
                for p in range(n):
                    best = sorted(entries(c.logprobs, p).items(), key=lambda kv: -kv[1])[:TOP_K]
                    ti[p, :len(best)] = [t for t, _ in best]
                    tl[p, :len(best)] = [v for _, v in best]
                top_rows.append(rid)
                top_ids.append(ti)
                top_lp.append(tl)
            records.append({
                "row_id": rid, "language": r.get("language"),
                "prompt_tokens": len(prompt["prompt_token_ids"]), "output_tokens": len(ids),
                "finish_reason": c.finish_reason, "stop_reason": c.stop_reason,
                "logprob_sum": float(np.nansum(lp)), "text": c.text,
            })
        ids_flat, offsets = concat([a for a, _ in own_lp], np.int32)
        lp_flat, _ = concat([b for _, b in own_lp], np.float32)
        tid_flat, toff = concat(top_ids, np.int32)
        tlp_flat, _ = concat(top_lp, np.float16)
        tmp = parts / f"gen_{k:03d}.tmp.npz"
        np.savez_compressed(tmp, records=np.array(json.dumps(records, ensure_ascii=False)), token_ids=ids_flat,
                            offsets=offsets, logprobs=lp_flat, top_rows=np.array(top_rows), top_ids=tid_flat,
                            top_offsets=toff, top_logprobs=tlp_flat, seconds=seconds)
        tmp.rename(part)
        n_tok = sum(r["output_tokens"] for r in records)
        print(f"chunk {k + 1}/{n_chunks}: {len(records)} rows, {n_tok:,} tokens in {seconds:.0f}s "
              f"({n_tok / max(seconds, 1e-9):,.0f} tok/s)", flush=True)
        progress(min((k + 1) * CHUNK_ROWS, len(rows)), len(rows), "rows generated")

    # the reference answer under the model, teacher forced: logprob of each of its tokens
    t1 = time.time()
    references = {}
    prompts, params, order = [], [], []
    for r in rows:
        rid = r["row_id"]
        p_ids = prompt_ids(tokenizer, r["messages"][:-1])
        r_ids = tokenizer.encode(r["messages"][-1]["content"], add_special_tokens=False)
        if not r_ids or len(p_ids) + len(r_ids) > max_len:
            references[rid] = {"ref_tokens": len(r_ids), "ref_logprob_sum": None}
            continue
        prompts.append(TokensPrompt(prompt_token_ids=p_ids + r_ids))
        params.append(score_params(r_ids, None, len(p_ids)))
        order.append((rid, r_ids))
    for out_ in llm.generate(prompts, params, use_tqdm=False) if prompts else []:
        rid, r_ids = order.pop(0)
        lp, _ = scored_logprobs(out_, r_ids, None)
        references[rid] = {"ref_tokens": len(r_ids), "ref_logprob_sum": float(np.nansum(lp))}
    ref_s = time.time() - t1
    print(f"reference NLL: {len(references)} rows in {ref_s:.0f}s", flush=True)

    # merge the chunks
    records, own_ids, own_lp, top_rows, top_ids, top_lp = [], [], [], [], [], []
    for k in range(n_chunks):
        z = np.load(parts / f"gen_{k:03d}.npz")
        chunk_records = json.loads(str(z["records"]))
        off = z["offsets"]
        for i, r in enumerate(chunk_records):
            records.append({**r, **references[r["row_id"]]})
            own_ids.append(z["token_ids"][off[i]:off[i + 1]])
            own_lp.append(z["logprobs"][off[i]:off[i + 1]])
        toff = z["top_offsets"]
        for i, rid in enumerate(z["top_rows"]):
            top_rows.append(str(rid))
            top_ids.append(z["top_ids"][toff[i]:toff[i + 1]])
            top_lp.append(z["top_logprobs"][toff[i]:toff[i + 1]])
    with open(out / "generations.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ids_flat, offsets = concat(own_ids, np.int32)
    lp_flat, _ = concat(own_lp, np.float32)
    np.savez_compressed(out / "tokens.npz", row_ids=np.array([r["row_id"] for r in records]), token_ids=ids_flat,
                        offsets=offsets, logprobs=lp_flat)
    tid_flat, toff = concat(top_ids, np.int32)
    tlp_flat, _ = concat(top_lp, np.float16)
    np.savez_compressed(out / "top.npz", row_ids=np.array(top_rows), offsets=toff, token_ids=tid_flat,
                        logprobs=tlp_flat, top_k=TOP_K)

    n_tok = sum(r["output_tokens"] for r in records)
    info = {
        "model": args.model, "model_dir": str(model_dir), "task": args.task, "rows": len(records),
        "limit_rows": args.limit_rows, "decoding": settings, "response_schema": schema, "max_tokens": max_tokens,
        "max_model_len": max_len, "enable_thinking": False, "seeds": "row_seed(row_id)", "top_k": TOP_K,
        "top_positions": TOP_POSITIONS, "analysis_rows": len(top_rows), "vllm": vllm_version,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_memory_utilization": args.gpu_memory_utilization,
        "load_s": round(load_s, 1), "generate_s": round(gen_s, 1), "reference_s": round(ref_s, 1),
        "generated_tokens": n_tok, "tokens_per_s": round(n_tok / gen_s, 1) if gen_s else None,
        "requests_per_s": round(len(records) / gen_s, 3) if gen_s else None,
        "finished": sum(r["finish_reason"] == "stop" for r in records), "complete": True,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "generation.json").write_text(json.dumps(info, indent=1))
    shutil.rmtree(parts)
    print("DONE " + json.dumps(info), flush=True)


# ----------------------------------------------------------------------------------------------------------- scoring

def score_params(target_ids: list[int], top: np.ndarray | None, prompt_len: int):
    """Teacher-forced scoring of `target_ids` after a prompt of `prompt_len` tokens: per target position, the
    logprob of the target token (column 0) and, when `top` is given, of the ids in `top` (-1 = padding)."""
    from vllm import SamplingParams

    ids = np.array(target_ids, dtype=np.int32)[:, None]
    if top is not None:
        pad = np.full((len(target_ids), top.shape[1]), -1, dtype=np.int32)
        pad[:len(top)] = top[:len(target_ids)]
        ids = np.concatenate([ids, pad], axis=1)
    return SamplingParams(max_tokens=1, temperature=0.0, prompt_logprob_token_ids=ids,
                          prompt_logprob_start=prompt_len - 1, flat_logprobs=True, detokenize=False)


def scored_logprobs(output, target_ids: list[int], top: np.ndarray | None):
    """(logprob of each target token, logprobs of the `top` ids at the positions that have them; NaN = padding).

    vLLM returns `prompt_token_id_logprobs` shaped [len(target_ids), 1 + top width]: row i scores its ids as
    predictions of target token i (needs vLLM's V2 model runner, the default for both models of the paper)."""
    scores = np.asarray(output.prompt_token_id_logprobs, dtype=np.float32)
    assert scores.shape[0] == len(target_ids), (scores.shape, len(target_ids))
    lp = scores[:, 0].copy()
    top_lp = None
    if top is not None:
        top_lp = scores[:len(top), 1:].copy()
        top_lp[top[:len(target_ids)] < 0] = np.nan
    return lp, top_lp


def cross(args):
    from vllm.inputs import TokensPrompt

    base_dir = Path(args.model_dir).expanduser()
    messages = {r["row_id"]: r["messages"] for r in load_rows(args.rows, args.limit_rows)}
    t0 = time.time()
    llm, max_len = make_llm(base_dir, args.gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()
    print(f"loaded {base_dir.name} in {time.time() - t0:.0f}s; scoring {', '.join(args.targets)}", flush=True)

    for n_target, target in enumerate(args.targets):
        tdir = Path(args.results) / target / args.task
        if (tdir / "cross.json").exists() and json.loads((tdir / "cross.json").read_text()).get("complete"):
            print(f"skip {target}: already scored", flush=True)
            continue
        t1 = time.time()
        tok = np.load(tdir / "tokens.npz")
        top = np.load(tdir / "top.npz")
        top_by_row = {str(r): top["token_ids"][top["offsets"][i]:top["offsets"][i + 1]]
                      for i, r in enumerate(top["row_ids"])}
        row_ids = [str(r) for r in tok["row_ids"]]
        lps, top_rows, top_lps = [], [], []
        for start in range(0, len(row_ids), CHUNK_ROWS):
            prompts, params, jobs = [], [], []
            for i in range(start, min(start + CHUNK_ROWS, len(row_ids))):
                rid = row_ids[i]
                ids = tok["token_ids"][tok["offsets"][i]:tok["offsets"][i + 1]].tolist()
                p_ids = prompt_ids(tokenizer, messages[rid][:-1])
                t = top_by_row.get(rid)
                if not ids or len(p_ids) + len(ids) > max_len:
                    jobs.append((rid, ids, t, None))
                    continue
                prompts.append(TokensPrompt(prompt_token_ids=p_ids + ids))
                params.append(score_params(ids, t, len(p_ids)))
                jobs.append((rid, ids, t, len(prompts) - 1))
            outputs = llm.generate(prompts, params, use_tqdm=False) if prompts else []
            for rid, ids, t, j in jobs:
                if j is None:
                    lp, tlp = np.full(len(ids), np.nan, np.float32), (None if t is None else np.full(t.shape, np.nan))
                else:
                    lp, tlp = scored_logprobs(outputs[j], ids, t)
                lps.append(lp)
                if t is not None:
                    top_rows.append(rid)
                    top_lps.append(tlp)
            progress(min(start + CHUNK_ROWS, len(row_ids)), len(row_ids), f"rows scored for {target}")
        lp_flat, offsets = concat(lps, np.float32)
        tlp_flat, toff = concat(top_lps, np.float16)
        np.savez_compressed(tdir / "cross.npz", row_ids=np.array(row_ids), offsets=offsets, logprobs=lp_flat,
                            top_rows=np.array(top_rows), top_offsets=toff, top_logprobs=tlp_flat)
        info = {"target": target, "scored_by": args.model, "scored_by_dir": str(base_dir), "task": args.task,
                "rows": len(row_ids), "seconds": round(time.time() - t1, 1), "complete": True,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (tdir / "cross.json").write_text(json.dumps(info, indent=1))
        print(f"DONE {target} ({n_target + 1}/{len(args.targets)}) " + json.dumps(info), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["generate", "cross"])
    p.add_argument("--model", required=True, help="name of the model's folder in --results")
    p.add_argument("--model-dir", required=True, help="checkpoint directory (cross: the base model)")
    p.add_argument("--task", required=True, help="name of the task's folder in --results")
    p.add_argument("--rows", required=True, help="JSON Lines file of the task's rows (module docstring)")
    p.add_argument("--results", default="results", help="results directory")
    p.add_argument("--decoding", choices=["greedy", "model_default"], default="greedy")
    p.add_argument("--schema", default=None, help="generate: JSON schema file that constrains the answer")
    p.add_argument("--targets", nargs="*", default=[], help="cross: the models whose answers to score")
    p.add_argument("--limit-rows", type=int, default=None, help="first N rows only (quick test)")
    p.add_argument("--max-tokens", type=int, default=None, help=f"answer cap (default {MAX_NEW_TOKENS})")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = p.parse_args()
    args.out_dir = str(Path(args.results) / args.model / args.task)
    generate(args) if args.mode == "generate" else cross(args)


if __name__ == "__main__":
    main()
