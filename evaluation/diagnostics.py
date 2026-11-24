"""The diagnostics of the REAP and REAM papers, as the paper computes them from the files that generate.py and
probe.py write.

    is_loop, tail_ratio        repetition loops: an answer that ran to the token cap, or whose last 3,000
                               characters compress below 10%
    distinct_n                 n-gram diversity (distinct-n), within answers and across a model's answers
    perplexity                 a model's perplexity of its own answers (tokens.npz) or the base model's perplexity
                               of a compressed model's answers (cross.npz): the cross-perplexity
    jsd_by_position, mean_jsd  completion-logit Jensen-Shannon divergence between a compressed model and its
                               base model, along the compressed model's answers
    ranks                      numerical and effective rank of the pre-logit embeddings (probe.npz)
    removed_share              share of the base model's routing that went to the experts a method removed

Conventions of the paper: answers that ran to the token cap are left out of the perplexities (a loop is easy to
predict and would pull every number down), loops are left out of distinct-n, and tasks decoded under a JSON schema
skip the first answer token in every log-probability measure (`skip=1`): the schema forces it, and the base model,
scoring without the schema, gives it a very low probability.
"""

import json
import re
import zlib
from pathlib import Path

import numpy as np

LOOP_MIN_CHARS, LOOP_MAX_RATIO = 1500, 0.10


# -------------------------------------------------------------------------------------------------- loops, diversity

def tail_ratio(text: str) -> float:
    """Compressed size / size of the last 3,000 characters (zlib, level 9)."""
    b = (text or "")[-3000:].encode()
    return len(zlib.compress(b, 9)) / max(1, len(b))


def is_loop(text: str, capped: bool = False) -> bool:
    """A repetition loop: the answer ran to the token cap (finish reason "length"), or it has at least 1,500
    characters and its last 3,000 compress below 10%."""
    return bool(capped) or (len(text or "") >= LOOP_MIN_CHARS and tail_ratio(text) < LOOP_MAX_RATIO)


def visible_text(text: str, html_field: str | None = None) -> str:
    """The text whose words are counted. With `html_field` (email templates: "html") the answer is JSON that
    carries an HTML document in that field: the HTML's text, without tags, styles and scripts."""
    if not html_field:
        return text or ""
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        s = json.loads(s).get(html_field) or s
    except Exception:
        pass
    s = re.sub(r"(?is)<(style|script|head)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"<[^>]+>|&[a-z#0-9]+;", " ", s)
    return s


def distinct_n(texts: list[str], n: int = 2, html_field: str | None = None) -> dict:
    """Distinct n-grams / n-grams over lower-cased words: `within` each answer (mean over answers) and `across`
    all answers pooled. Pass the answers that are not loops."""
    toks = [re.findall(r"\w+", visible_text(t, html_field).lower()) for t in texts]
    grams = [list(zip(*(t[i:] for i in range(n)))) for t in toks]
    within = [len(set(g)) / len(g) for g in grams if g]
    pooled = [x for g in grams for x in g]
    return {"within": float(np.mean(within)) if within else float("nan"),
            "across": len(set(pooled)) / len(pooled) if pooled else float("nan")}


# -------------------------------------------------------------------------------------------------------- perplexity

def row_logprobs(path, skip: int = 0) -> dict[str, tuple[float, int]]:
    """row id -> (sum of the answer-token log-probabilities, tokens counted), from tokens.npz (a model's own) or
    cross.npz (the base model's log-probabilities of a compressed model's answer tokens)."""
    z = np.load(path)
    off, lp = z["offsets"], np.nan_to_num(z["logprobs"].astype(np.float64), nan=0.0)
    csum = np.concatenate([[0.0], np.cumsum(lp)])
    start = np.minimum(off[:-1] + skip, off[1:])
    sums, tokens = csum[off[1:]] - csum[start], off[1:] - start
    return {str(r): (float(s), int(t)) for r, s, t in zip(z["row_ids"], sums, tokens)}


def perplexity(path, rows=None, skip: int = 0) -> float:
    """Token-weighted perplexity over `rows` (default: all rows of the file). For the paper's numbers, `rows` are
    the answers that did not run to the token cap."""
    lp = row_logprobs(path, skip)
    keep = lp if rows is None else [r for r in rows if r in lp]
    total, tokens = sum(lp[r][0] for r in keep), sum(lp[r][1] for r in keep)
    return float(np.exp(-total / max(tokens, 1)))


# --------------------------------------------------------------------------------------------------------------- JSD

def jsd_by_position(top_path, cross_path, max_pos: int = 2048, skip: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(mean JSD in bits per answer position, rows that reach the position), on the analysis rows.

    Per position: the compressed model's top-20 distribution (top.npz) and the base model's probabilities of the
    same 20 tokens (cross.npz), each with the remaining mass as a 21st outcome. This is a lower bound of the JSD
    over the whole vocabulary."""
    tz, cz = np.load(top_path), np.load(cross_path)
    top_idx = {str(r): (tz["offsets"][i], tz["offsets"][i + 1]) for i, r in enumerate(tz["row_ids"])}
    cross_idx = {str(r): (cz["top_offsets"][i], cz["top_offsets"][i + 1]) for i, r in enumerate(cz["top_rows"])}
    ids, lp_c, lp_b = tz["token_ids"], tz["logprobs"].astype(np.float64), cz["top_logprobs"].astype(np.float64)
    sums, counts = np.zeros(max_pos), np.zeros(max_pos)

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a > 0, a * np.log2(a / np.where(b > 0, b, 1)), 0.0).sum(1)

    for rid, (s, e) in top_idx.items():
        if rid not in cross_idx:
            continue
        cs, ce = cross_idx[rid]
        n = min(e - s, ce - cs, max_pos)
        pad = ids[s:s + n] < 0
        p = np.where(pad, 0.0, np.exp(np.nan_to_num(lp_c[s:s + n], nan=-np.inf)))
        q = np.where(pad, 0.0, np.exp(np.nan_to_num(lp_b[cs:cs + n], nan=-np.inf)))
        p = np.concatenate([p, np.clip(1 - p.sum(1, keepdims=True), 0, 1)], axis=1)
        q = np.concatenate([q, np.clip(1 - q.sum(1, keepdims=True), 0, 1)], axis=1)
        mix = (p + q) / 2
        sums[:n] += np.clip(0.5 * kl(p, mix) + 0.5 * kl(q, mix), 0, 1)
        counts[:n] += 1
    counts[:skip] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(counts > 0, sums / counts, np.nan), counts


def mean_jsd(top_path, cross_path, max_pos: int = 2048, skip: int = 0) -> float:
    """The paper's JSD of a model on a task: the mean over positions, weighted by the rows that reach each."""
    jsd, counts = jsd_by_position(top_path, cross_path, max_pos, skip)
    return float(np.average(jsd[counts > 0], weights=counts[counts > 0])) if (counts > 0).any() else float("nan")


# -------------------------------------------------------------------------------------------------------------- rank

def ranks(singular_values, n_embeddings: int, dim: int) -> dict:
    """Numerical rank (singular values above s_max * max(n, d) * float32 epsilon, the torch and numpy default) and
    effective rank (the exponential of the entropy of the normalised singular values). The paper reports the
    numerical rank of the raw embeddings (`sv__<task>` of probe.npz) and the effective rank of the mean-centred
    ones (`sv_centered__<task>`)."""
    s = np.asarray(singular_values, np.float64)
    tol = s.max() * max(n_embeddings, dim) * np.finfo(np.float32).eps
    p = s / s.sum()
    return {"numerical_rank": int((s > tol).sum()), "effective_rank": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))}


# ----------------------------------------------------------------------------------------------------------- routing

def removed_share(route: np.ndarray, retained: list[list[int]], route_layers: list[str] | None = None,
                  report_layers: list[str] | None = None) -> float:
    """Share (%) of the base model's routing that went to the experts a compression run removed, mean over the
    MoE layers. `route` is [layers, experts]: per layer and expert, the summed gate weight of the tokens of
    interest (a `route__<task>__<language group>__<prompt|answer>` array of probe.npz, or a sum of several).
    `retained` lists, per layer, the experts the run kept (`retained` of the report compress.py writes). With the
    layer names of both (`layers` of probe.json and of the report) the layers are matched by name; without them
    the two must list the layers in the same order."""
    route = np.asarray(route, np.float64)
    if route_layers is not None and report_layers is not None:
        pos = {name: i for i, name in enumerate(route_layers)}
        pairs = [(route[pos[name]], retained[li]) for li, name in enumerate(report_layers) if name in pos]
    else:
        pairs = list(zip(route, retained))
    shares = []
    for row, kept in pairs:
        keep = np.zeros(len(row), bool)
        keep[kept] = True
        if row.sum() > 0:
            shares.append(row[~keep].sum() / row.sum())
    return float(np.mean(shares)) * 100 if shares else float("nan")


def load_route(probe_dir, task: str, segment: str, groups: list[str] | None = None) -> np.ndarray | None:
    """Sum of the routing arrays of one task and segment ("prompt" or "answer") over the language groups given
    (default: all in the file)."""
    z = np.load(Path(probe_dir) / "probe.npz")
    prefix = f"route__{task}__"
    keys = [k for k in z.files if k.startswith(prefix) and k.endswith(f"__{segment}")
            and (groups is None or k[len(prefix):-len(segment) - 2] in groups)]
    return np.sum([z[k] for k in keys], axis=0) if keys else None
