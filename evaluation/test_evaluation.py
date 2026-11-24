"""Checks of the evaluation code on made-up data; they run on the CPU.

    python evaluation/test_evaluation.py

  1. stats         McNemar's exact test, the sign-flip permutation test, Holm's correction and the paired bootstrap
                   on arrays with known answers
  2. checks        the exact-match scores, the rules of domain descriptions and the two relaxed gates on small
                   examples
  3. diagnostics   the loop rule, distinct-n, the ranks, the routing share, and perplexity and JSD on arrays with
                   known answers
  4. generate.py   both modes end to end on example/rows.jsonl with a stand-in for vLLM (FakeLLM below): a "model"
                   whose next-token probabilities are a known function. This checks how the script reads its rows
                   and writes its files, and that diagnostics.py reads them; it does not check vLLM.
  5. probe.py      end to end on a tiny random Qwen3-MoE model on the CPU; skipped without PyTorch and Transformers.
"""

import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checks  # noqa: E402
import diagnostics  # noqa: E402
import stats  # noqa: E402


def ok(name: str):
    print(f"ok   {name}", flush=True)


# ------------------------------------------------------------------------------------------------------------ 1. stats

def test_stats():
    from scipy.stats import binomtest

    # 30 rows only x passes, 12 rows only y passes, 58 both: McNemar is a binomial test on the 42 discordant rows
    x = np.array([100.0] * 30 + [0.0] * 12 + [100.0] * 58)
    y = np.array([0.0] * 30 + [100.0] * 12 + [100.0] * 58)
    r = stats.compare(x, y, binary=True, seed=3)
    assert r["test"] == "McNemar exact" and math.isclose(r["p"], binomtest(12, 42, 0.5).pvalue)
    assert math.isclose(r["diff"], -18.0) and r["diff_lo"] < -18.0 < r["diff_hi"] < 0
    assert (r["rows_x_higher"], r["rows_y_higher"]) == (30, 12)
    assert math.isclose(r["retention"], 70 / 88 * 100) and r["retention_lo"] < r["retention"] < r["retention_hi"]
    assert r == stats.compare(x, y, binary=True, seed=3), "same seed, same interval"

    same = stats.compare(x, x, binary=True)
    assert same["p"] == 1.0 and same["diff_lo"] == same["diff_hi"] == 0.0
    assert stats.compare(x, x, binary=False)["p"] == 1.0

    # scores: a shift of 5 points with little noise is significant, pure noise is not
    rng = np.random.default_rng(0)
    base = rng.uniform(40, 90, 400)
    shifted = stats.compare(base, base + 5 + rng.normal(0, 3, 400), binary=False)
    assert shifted["test"] == "sign-flip permutation" and math.isclose(shifted["p"], 1 / (stats.B + 1))
    assert stats.compare(base, base + rng.normal(0, 3, 400), binary=False)["p"] > 0.05

    # Holm by hand: sorted 0.01, 0.03, 0.04 -> 0.03, 0.06, max(0.06, 0.04)
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(stats.holm([0.5, 0.9]), [1.0, 1.0])

    lo, hi = stats.mean_retention([stats.paired_bootstrap(x, y, seed=0)[1], stats.paired_bootstrap(x, x, seed=1)[1]])
    assert lo < (70 / 88 * 100 + 100) / 2 < hi

    out = subprocess.run([sys.executable, str(HERE / "stats.py"), "--scores", str(HERE / "example" / "scores.jsonl"),
                          "--binary", "--pairs", "base:pruned", "base:merged", "merged:pruned"],
                         capture_output=True, text=True, check=True).stdout
    assert "Holm correction over these 3 comparisons" in out and "base:merged" in out
    ok("stats: McNemar, sign-flip permutation, Holm, bootstrap, command line")


# ----------------------------------------------------------------------------------------------------------- 2. checks

def test_checks():
    assert checks.parse_json('```json\n{"a": 1}\n```') == {"a": 1} and checks.parse_json("not json") is None

    ref = '{"isColorMentioned": true, "isFontMentioned": false}'
    assert checks.score_website_cues(ref, ref) == {"valid": True, "pred": {"isColorMentioned": True, "isFontMentioned": False},
                                                    "correct": True}
    assert not checks.score_website_cues('{"isColorMentioned": true, "isFontMentioned": true}', ref)["correct"]
    bad = checks.score_website_cues('{"isColorMentioned": "yes"}', ref)
    assert not bad["valid"] and not bad["correct"]

    ref = '{"brand_name": "Nordwind", "source": "detected"}'
    assert checks.score_brand(ref, ref)["correct"]
    wrong_case = checks.score_brand('{"brand_name": "nordwind", "source": "detected"}', ref)
    assert not wrong_case["correct"] and wrong_case["lenient_correct"] and wrong_case["detected"]
    generated = checks.score_brand('{"brand_name": "Nordwind", "source": "generated"}', ref)
    assert not generated["correct"] and not generated["detected"]  # the model did not find the brand in the text

    ref = '{"category": "Garden Tools"}'
    assert checks.score_categorisation('{"category": " garden  tools"}', ref)["exact"]
    assert not checks.score_categorisation('{"category": "Tools"}', ref)["exact"]
    assert not checks.score_categorisation("Garden Tools", ref)["valid"]

    good = checks.domain_checks("Este dominio es ideal para una tienda de bicicletas. Es corto y fácil de recordar.",
                                "es", "bicis.example", detect=lambda t: "es")
    assert good["sentences"] == 2 and good["max_words_per_sentence"] == 9 and not checks.breaks_rule(good)
    assert checks.breaks_rule(good, capped=True)
    opener = checks.domain_checks("This domain es ideal para una tienda.", "es", "bicis.example", detect=lambda t: "es")
    assert opener["english_opening"] and checks.breaks_rule(opener)
    assert not checks.domain_checks("This domain is short.", "en", "bikes.example")["english_opening"]
    repeats = checks.domain_checks("Bicis.example es un dominio corto.", "es", "bicis.example")
    assert repeats["repeats_domain"] and repeats["language_ok"] is None and checks.breaks_rule(repeats)
    wrong = checks.domain_checks("A short domain for a bike shop.", "es", "bicis.example", detect=lambda t: "en")
    assert wrong["language_ok"] is False and checks.breaks_rule(wrong)
    three = checks.domain_checks("Uno. Dos. Tres.", "es", "x.example")
    assert three["sentences"] == 3 and checks.breaks_rule(three)
    long = checks.domain_checks(" ".join(["palabra"] * 21) + ".", "es", "x.example")
    assert long["max_words_per_sentence"] == 21 and checks.breaks_rule(long)

    assert checks.relaxed_html_complete("<body><table></table></body>")
    assert not checks.relaxed_html_complete("<table><tr><td>Hi</td></tr></table>")  # a bare fragment still fails
    assert not checks.relaxed_html_complete("<body><table>")  # truncated
    assert checks.relaxed_unsubscribe('<p><a href="#">Unsubscribe</a></p>')
    assert checks.relaxed_unsubscribe('<a href="{{unsubscribe-link}}">here</a>')
    assert checks.relaxed_unsubscribe('<a href="#">Se désabonner</a>') and checks.relaxed_unsubscribe('<a href="#">Atsisakyti</a>')
    assert not checks.relaxed_unsubscribe('<p>Unsubscribe</p><a href="/shop">Shop</a>')  # no unsubscribe link
    ok("checks: exact-match scores, domain-description rules, relaxed gates")


# ------------------------------------------------------------------------------------------------------ 3. diagnostics

def test_diagnostics():
    words = ("The quick brown fox jumps over the lazy dog while seven wizards quietly judge every boxing match and "
             "nobody explains why the zebra keeps a violin under its hat. ")
    rng = np.random.default_rng(0)
    varied = " ".join("".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), rng.integers(2, 9))) for _ in range(600))
    assert not diagnostics.is_loop(varied) and diagnostics.is_loop(varied, capped=True)
    assert diagnostics.is_loop(varied + "again and again " * 400)  # ends in a loop
    assert not diagnostics.is_loop("yes " * 200)  # repeats, but shorter than 1,500 characters
    assert diagnostics.tail_ratio(words * 40) < 0.10

    d = diagnostics.distinct_n(["a b a b", "a b c d"], 2)
    assert math.isclose(d["within"], (2 / 3 + 1) / 2) and math.isclose(d["across"], 4 / 6)
    html = json.dumps({"html": "<html><head><style>p {color: red}</style></head><body><p>Hello&nbsp;there</p></body></html>"})
    assert diagnostics.visible_text(html, "html").split() == ["Hello", "there"]

    r = diagnostics.ranks(np.ones(16), 64, 16)
    assert r["numerical_rank"] == 16 and math.isclose(r["effective_rank"], 16)
    r = diagnostics.ranks([1.0, 1.0, 1e-12, 0.0], 64, 4)
    assert r["numerical_rank"] == 2 and math.isclose(r["effective_rank"], 2, rel_tol=1e-6)

    route = np.array([[4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    assert math.isclose(diagnostics.removed_share(route, [[0, 1], [0, 1, 2]]), (0.3 + 0.25) / 2 * 100)
    assert math.isclose(diagnostics.removed_share(route, [[0, 1, 2], [0, 1]], ["L0", "L1"], ["L1", "L0"]), (0.3 + 0.25) / 2 * 100)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # two rows of 3 and 2 tokens, each token with probability 1/4: perplexity 4, also without the first tokens
        np.savez(tmp / "tokens.npz", row_ids=np.array(["a", "b"]), offsets=np.array([0, 3, 5]),
                 logprobs=np.full(5, math.log(0.25), np.float32))
        assert math.isclose(diagnostics.perplexity(tmp / "tokens.npz"), 4, rel_tol=1e-6)
        assert math.isclose(diagnostics.perplexity(tmp / "tokens.npz", rows=["a"], skip=1), 4, rel_tol=1e-6)
        assert diagnostics.row_logprobs(tmp / "tokens.npz", skip=1)["b"][1] == 1
        # one analysis row, two positions, two tokens kept: equal distributions (JSD 0), then disjoint ones (1 bit)
        np.savez(tmp / "top.npz", row_ids=np.array(["a"]), offsets=np.array([0, 2]),
                 token_ids=np.array([[5, 6], [5, 6]], np.int32),
                 logprobs=np.log(np.array([[0.5, 0.5], [1.0, 1e-30]], np.float64)))
        np.savez(tmp / "cross.npz", top_rows=np.array(["a"]), top_offsets=np.array([0, 2]),
                 top_logprobs=np.log(np.array([[0.5, 0.5], [1e-30, 1.0]], np.float64)))
        jsd, counts = diagnostics.jsd_by_position(tmp / "top.npz", tmp / "cross.npz", max_pos=4)
        assert np.allclose(jsd[:2], [0.0, 1.0]) and list(counts) == [1, 1, 0, 0] and np.isnan(jsd[2])
        assert math.isclose(diagnostics.mean_jsd(tmp / "top.npz", tmp / "cross.npz", max_pos=4), 0.5)
        assert math.isclose(diagnostics.mean_jsd(tmp / "top.npz", tmp / "cross.npz", max_pos=4, skip=1), 1.0)
    ok("diagnostics: loops, distinct-n, ranks, routing share, perplexity, JSD")


# ------------------------------------------------------------------------------------------------------ 4. generate.py
# A stand-in for vLLM. A FakeLLM is a "model" with a number (its seed): at answer position p after a prompt with
# token ids `prompt`, it gives probability 0.5 to its preferred token, (sum(prompt) + 7 p + seed) mod 50, and 0.01 to
# every other token; it generates its preferred tokens and stops after ANSWER_TOKENS of them. So a model's
# perplexity of its own answers is 2, another model's perplexity of them is 100, and the JSD between a model and
# itself is 0.

ANSWER_TOKENS = 6


def _preferred(prompt: list[int], position: int, seed: int) -> int:
    return (sum(prompt) + 7 * position + seed) % 50


class _Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=False, tokenize=True,
                            return_dict=True):
        assert add_generation_prompt and not enable_thinking and tokenize and return_dict
        return {"input_ids": [ord(c) % 997 for m in messages for c in m["role"] + m["content"]]}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 997 for c in text]


def _fake_vllm() -> types.ModuleType:
    class Params:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class FlatLogprobs:
        def __init__(self, per_position: list[dict[int, float]]):
            self.start_indices, self.end_indices, self.token_ids, self.logprobs = [], [], [], []
            for entries in per_position:
                self.start_indices.append(len(self.token_ids))
                self.token_ids += list(entries)
                self.logprobs += list(entries.values())
                self.end_indices.append(len(self.token_ids))

    class LLM:
        def __init__(self, model, max_model_len, gpu_memory_utilization, seed, max_logprobs):
            self.seed = json.loads((Path(model) / "fake_model.json").read_text())["seed"]

        def get_tokenizer(self):
            return _Tokenizer()

        def generate(self, prompts, params, use_tqdm=False):
            outputs = []
            for prompt, p in zip(prompts, params):
                ids = list(prompt["prompt_token_ids"])
                if getattr(p, "prompt_logprob_token_ids", None) is not None:  # teacher-forced scoring
                    start = p.prompt_logprob_start + 1  # the prompt's length; the rest are the target tokens
                    assert p.max_tokens == 1 and len(ids) - start == len(p.prompt_logprob_token_ids)
                    scores = [[math.log(0.5 if t == _preferred(ids[:start], pos, self.seed) else 0.01) for t in row]
                              for pos, row in enumerate(p.prompt_logprob_token_ids)]
                    outputs.append(types.SimpleNamespace(prompt_token_id_logprobs=scores))
                    continue
                n = min(ANSWER_TOKENS, p.max_tokens)
                tokens = [_preferred(ids, pos, self.seed) for pos in range(n)]
                per_position = []
                for t in tokens:
                    entries = {t: math.log(0.5)}
                    for other in range(1, p.logprobs):  # logprobs = 0: only the generated token's
                        entries[(t + other) % 50 + 100] = math.log(0.01)
                    per_position.append(entries)
                completion = types.SimpleNamespace(token_ids=tokens, logprobs=FlatLogprobs(per_position),
                                                   finish_reason="stop" if n == ANSWER_TOKENS else "length",
                                                   stop_reason=None, text=" ".join(map(str, tokens)))
                outputs.append(types.SimpleNamespace(outputs=[completion]))
            return outputs

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams, vllm.__version__ = LLM, Params, "stand-in"
    inputs, sampling = types.ModuleType("vllm.inputs"), types.ModuleType("vllm.sampling_params")
    inputs.TokensPrompt = dict
    sampling.StructuredOutputsParams = Params
    vllm.inputs, vllm.sampling_params = inputs, sampling
    return vllm, inputs, sampling


def test_generate():
    import generate

    vllm, inputs, sampling = _fake_vllm()
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.inputs", "vllm.sampling_params")}
    sys.modules.update({"vllm": vllm, "vllm.inputs": inputs, "vllm.sampling_params": sampling})
    rows_file = HERE / "example" / "rows.jsonl"
    rows = generate.load_rows(str(rows_file), None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, seed in (("base", 1), ("pruned", 2)):
                d = tmp / "models" / name
                d.mkdir(parents=True)
                (d / "config.json").write_text(json.dumps({"max_position_embeddings": 4096}))
                (d / "generation_config.json").write_text(json.dumps({"temperature": 0.7, "top_p": 0.8, "top_k": 20}))
                (d / "fake_model.json").write_text(json.dumps({"seed": seed}))

            def run(mode, model, **kw):
                args = dict(mode=mode, model=model, model_dir=str(tmp / "models" / model), task="example",
                            rows=str(rows_file), results=str(tmp / "results"), decoding="greedy", schema=None,
                            targets=[], limit_rows=None, max_tokens=None, gpu_memory_utilization=0.9)
                args.update(kw)
                args["out_dir"] = str(Path(args["results"]) / model / args["task"])
                with contextlib.redirect_stdout(io.StringIO()):  # the script's progress lines
                    (generate.generate if mode == "generate" else generate.cross)(types.SimpleNamespace(**args))
                return Path(args["out_dir"])

            base = run("generate", "base")
            pruned = run("generate", "pruned", decoding="model_default", schema=str(HERE / "example" / "schema.json"))
            run("cross", "base", targets=["pruned", "base"])

            records = [json.loads(line) for line in (pruned / "generations.jsonl").read_text().splitlines()]
            assert [r["row_id"] for r in records] == [r["row_id"] for r in rows]
            assert all(r["output_tokens"] == ANSWER_TOKENS and r["finish_reason"] == "stop" for r in records)
            assert all(math.isclose(r["logprob_sum"], ANSWER_TOKENS * math.log(0.5), rel_tol=1e-5) for r in records)
            assert all(r["ref_tokens"] == len(row["messages"][-1]["content"]) for r, row in zip(records, rows))
            info = json.loads((pruned / "generation.json").read_text())
            assert info["complete"] and info["decoding"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
            assert info["max_tokens"] == generate.SCHEMA_MAX_TOKENS and info["response_schema"]["required"] == ["category"]
            assert json.loads((base / "generation.json").read_text())["decoding"] == {"temperature": 0.0}
            assert not (pruned / "parts").exists()
            top = np.load(pruned / "top.npz")
            assert len(top["row_ids"]) == len(rows) and top["token_ids"].shape == (len(rows) * ANSWER_TOKENS, generate.TOP_K)

            ids = [r["row_id"] for r in rows]
            assert math.isclose(diagnostics.perplexity(pruned / "tokens.npz", ids), 2, rel_tol=1e-5)  # own
            assert math.isclose(diagnostics.perplexity(pruned / "cross.npz", ids), 100, rel_tol=1e-5)  # the base model's
            assert math.isclose(diagnostics.perplexity(base / "cross.npz", ids), 2, rel_tol=1e-5)
            assert abs(diagnostics.mean_jsd(base / "top.npz", base / "cross.npz")) < 1e-6
            assert diagnostics.mean_jsd(pruned / "top.npz", pruned / "cross.npz") > 0.1
            assert math.isclose(diagnostics.perplexity(pruned / "tokens.npz", ids, skip=1), 2, rel_tol=1e-5)

            # the seed and the analysis rows depend on the row id only
            assert generate.row_seed("row-000") == generate.row_seed("row-000") != generate.row_seed("row-001")
            assert generate.analysis_rows(ids, 5) == generate.analysis_rows(ids[::-1], 5) and len(generate.analysis_rows(ids, 5)) == 5
    finally:
        for k, v in saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)
    ok("generate.py: generate and cross with a stand-in for vLLM; diagnostics read its files")


# --------------------------------------------------------------------------------------------------------- 5. probe.py

def test_probe():
    try:
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM
    except ImportError as e:
        print(f"skip probe.py: {e}", flush=True)
        return
    rows = [json.loads(line) for line in (HERE / "example" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # a character-level tokenizer with a minimal chat template, and a random 2-layer model with 8 experts
        chars = sorted({c for r in rows for m in r["messages"] for c in m["role"] + m["content"]} | set("<|>\n"))
        tok = Tokenizer(models.WordLevel({c: i for i, c in enumerate(["[UNK]"] + chars)}, unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
        fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
        fast.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
                              "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")
        fast.save_pretrained(tmp / "model")
        torch.manual_seed(0)
        config = Qwen3MoeConfig(vocab_size=len(chars) + 1, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=2, moe_intermediate_size=16, num_experts=8,
                                num_experts_per_tok=2, decoder_sparse_step=1, max_position_embeddings=512)
        Qwen3MoeForCausalLM(config).save_pretrained(tmp / "model")
        subprocess.run([sys.executable, str(HERE / "probe.py"), "--model", "tiny", "--model-dir", str(tmp / "model"),
                        "--out", str(tmp / "probe"), "--rows", f"example={HERE / 'example' / 'rows.jsonl'}",
                        "--analysis-rows", "0", "--per-row", "16", "--routing", "--device", "cpu"], check=True,
                       capture_output=True, text=True)
        info, z = json.loads((tmp / "probe" / "probe.json").read_text()), np.load(tmp / "probe" / "probe.npz")
        assert info["complete"] and info["rows_per_task"] == {"example": len(rows)} and len(info["layers"]) == 2
        t = info["tokens"]["example"]
        assert t["dim"] == 32 and t["embeddings"] == len(rows) * 16
        r = diagnostics.ranks(z["sv_centered__example"], t["embeddings"], t["dim"])
        assert 1 < r["effective_rank"] <= 32 and diagnostics.ranks(z["sv__example"], t["embeddings"], t["dim"])["numerical_rank"] <= 32
        # every token's routing sums to one per layer: the prompt and answer arrays together count the tokens read
        total = sum(z[k] for k in z.files if k.startswith("route__example__"))
        assert total.shape == (2, 8) and np.allclose(total.sum(1), t["read"])
        groups = {k.split("__")[2] for k in z.files if k.startswith("route__")}
        assert groups == {"en", "es", "fr", "pt", "id", "other"}
        answer = diagnostics.load_route(tmp / "probe", "example", "answer")
        share = diagnostics.removed_share(answer, [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
        assert 0 <= share <= 100
        assert math.isclose(diagnostics.removed_share(answer, [list(range(8))] * 2), 0.0)
    ok("probe.py: embeddings, ranks and routing on a tiny random Qwen3-MoE model (CPU)")


if __name__ == "__main__":
    test_stats()
    test_checks()
    test_diagnostics()
    test_generate()
    test_probe()
    print("all checks passed")
