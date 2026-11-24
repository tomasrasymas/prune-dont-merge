"""Paired statistics of the paper: every confidence interval and p-value comes from these functions.

Every comparison is paired: two models answer the same rows, and `x[i]` and `y[i]` are their scores on row i
(0 or 100 for a pass/fail metric, a rubric score for email templates). `y - x` is reported, so `x` is the base
model when a compressed model is compared with it, and REAM when REAP is compared with REAM.

    paired_bootstrap   resampled mean differences and ratios of means (10,000 resamples of rows)
    p_value            McNemar's exact test for pass/fail metrics, a paired sign-flip permutation test on the
                       mean for scores
    holm               Holm's correction over one family of comparisons
    compare            all of the above for one pair, as a dict

Seeds, for reproduction of the paper's intervals: the bootstrap of a comparison with the base model used the
index of the task (0 to 4) as seed, the bootstrap of a REAP - REAM comparison 500 plus that index. The permutation
test always uses seed 12345.

Command line: comparisons between models of a scores file, Holm-corrected over the pairs given.

    python stats.py --scores example/scores.jsonl --binary --pairs base:pruned base:merged merged:pruned

The file has one JSON object per line, {"model": ..., "row_id": ..., "value": ...}; a pair is x:y and y - x is
reported on the rows both models have.
"""

import argparse
import json

import numpy as np

B = 10_000  # bootstrap and permutation resamples


def paired_bootstrap(x: np.ndarray, y: np.ndarray, n_boot: int = B, seed: int = 0):
    """Resampled mean differences (y - x) and ratios (mean y / mean x) over rows, paired."""
    rng = np.random.default_rng(seed)
    n, diffs, ratios = len(x), [], []
    for start in range(0, n_boot, 1000):
        idx = rng.integers(0, n, (min(1000, n_boot - start), n))
        mx, my = x[idx].mean(1), y[idx].mean(1)
        diffs.append(my - mx)
        ratios.append(np.divide(my, mx, out=np.full_like(my, np.nan), where=mx != 0))
    return np.concatenate(diffs), np.concatenate(ratios)


def p_value(x: np.ndarray, y: np.ndarray, binary: bool) -> tuple[float, str]:
    """Two-sided p-value of mean(y) = mean(x) on paired rows, and the name of the test."""
    from scipy import stats

    if binary:
        # McNemar's exact test: a binomial test on the rows only one of the two models passes
        b, c = int(((x > 0) & (y == 0)).sum()), int(((x == 0) & (y > 0)).sum())
        return (1.0 if b + c == 0 else stats.binomtest(min(b, c), b + c, 0.5).pvalue), "McNemar exact"
    # scores: a paired sign-flip permutation test on the mean difference, the quantity the bootstrap interval is
    # for (a Wilcoxon signed-rank test would test a median-type shift instead)
    d = np.asarray(y, float) - np.asarray(x, float)
    if np.allclose(d, 0):
        return 1.0, "sign-flip permutation"
    rng = np.random.default_rng(12345)
    obs = abs(d.mean())
    perm = np.abs((d[None, :] * rng.choice([-1.0, 1.0], size=(B, len(d)))).mean(axis=1))
    return float((np.sum(perm >= obs - 1e-12) + 1) / (B + 1)), "sign-flip permutation"


def holm(p) -> np.ndarray:
    """Holm-corrected p-values of one family, in the order given."""
    p = np.asarray(p, float)
    out, running = np.empty(len(p)), 0.0
    for i, k in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (len(p) - i) * p[k]))
        out[k] = running
    return out


def compare(x, y, binary: bool, seed: int = 0, n_boot: int = B) -> dict:
    """y against x on the same rows: means, difference and retention (mean y / mean x, %) with 95% paired
    bootstrap intervals, the rows on which each side scores higher, and the uncorrected p-value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    assert x.shape == y.shape and x.ndim == 1 and len(x) > 0, "x and y are the scores of the same rows"
    diffs, ratios = paired_bootstrap(x, y, n_boot, seed)
    p, test = p_value(x, y, binary)
    return {
        "n": len(x), "x": x.mean(), "y": y.mean(), "diff": y.mean() - x.mean(),
        "diff_lo": np.percentile(diffs, 2.5), "diff_hi": np.percentile(diffs, 97.5),
        "retention": y.mean() / x.mean() * 100 if x.mean() else float("nan"),
        "retention_lo": np.nanpercentile(ratios, 2.5) * 100, "retention_hi": np.nanpercentile(ratios, 97.5) * 100,
        "rows_y_higher": int((y > x).sum()), "rows_x_higher": int((x > y).sum()), "p": p, "test": test,
    }


def mean_retention(ratio_samples: list[np.ndarray]) -> tuple[float, float]:
    """95% interval (%) of the mean retention over several tasks, from each task's resampled ratios
    (`paired_bootstrap(...)[1]`, tasks resampled independently): the paper's retention over the discriminative
    and over the generative tasks."""
    samples = np.mean(ratio_samples, axis=0) * 100
    return float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", required=True, help='JSON Lines: {"model", "row_id", "value"} per line')
    ap.add_argument("--pairs", required=True, nargs="+", help="x:y model pairs; y - x is reported")
    ap.add_argument("--binary", action="store_true", help="pass/fail metric (values 0 and 100): McNemar's exact test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    values: dict[str, dict[str, float]] = {}
    with open(args.scores, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                values.setdefault(r["model"], {})[str(r["row_id"])] = float(r["value"])
    results = []
    for pair in args.pairs:
        a, b = pair.split(":")
        rows = sorted(set(values[a]) & set(values[b]))
        results.append(compare([values[a][r] for r in rows], [values[b][r] for r in rows], args.binary, args.seed))
    corrected = holm([r["p"] for r in results])
    print(f"{'pair (x:y)':<24}{'rows':>6}{'x':>8}{'y':>8}{'y - x [95% CI]':>26}{'retention % [95% CI]':>28}"
          f"{'p':>10}{'Holm p':>10}")
    for pair, r, ph in zip(args.pairs, results, corrected):
        diff = f"{r['diff']:+.1f} [{r['diff_lo']:+.1f}, {r['diff_hi']:+.1f}]"
        ret = f"{r['retention']:.1f} [{r['retention_lo']:.1f}, {r['retention_hi']:.1f}]"
        print(f"{pair:<24}{r['n']:>6}{r['x']:>8.1f}{r['y']:>8.1f}{diff:>26}{ret:>28}{r['p']:>10.2g}{ph:>10.2g}")
    print(f"test: {results[0]['test']}; Holm correction over these {len(results)} comparisons")


if __name__ == "__main__":
    main()
