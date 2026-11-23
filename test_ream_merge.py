"""Unit checks of the REAM port (ream_merge.py) without loading a model. They run on the CPU.

    python test_ream_merge.py --ream-repo <clone of SamsungSAILMontreal/ream at 84a3030>

Without --ream-repo, checks 1 and 2 (parity with the original code) are skipped.

1. Grouping parity: the port's pseudo_group (router-logit similarity from a Gram matrix, vectorized cosine)
   gives the same groups as the original ream.ream.pseudo_group on the same statistics.
2. PCA parity with the original weight_utils.pca_reduce.
3. Neuron alignment: an expert whose neurons are a shuffled copy of its centre is matched back exactly, and
   merging it into the centre returns the centre; this also checks permute_neurons for gated (Qwen, GLM)
   and non-gated relu2 (Nemotron) experts.
4. Token reservoir: all experts of a layer keep activations of the same tokens, sampled without repeats.
"""

import argparse
import sys

import numpy as np
import torch

from ream_merge import (
    REAMMergingModifier,
    _LayerStats,
    alignment_permutation,
    dist2sim,
    gate_distance_from_gram,
    merge_group,
    neuron_weight_matrix,
    pca_reduce,
    permute_neurons,
    pseudo_group,
)


class Gated(torch.nn.Module):
    def __init__(self, h, d):
        super().__init__()
        self.gate_proj, self.up_proj = torch.nn.Linear(h, d, bias=False), torch.nn.Linear(h, d, bias=False)
        self.down_proj = torch.nn.Linear(d, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Relu2(torch.nn.Module):
    def __init__(self, h, d):
        super().__init__()
        self.up_proj, self.down_proj = torch.nn.Linear(h, d, bias=False), torch.nn.Linear(d, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)) ** 2)


def check_grouping(ream_repo):
    sys.path.insert(0, ream_repo)
    from ream.ream import pseudo_group as orig_pseudo_group
    from ream.weight_utils import pca_reduce as orig_pca_reduce

    torch.manual_seed(0)
    for n_experts, k, c in [(128, 96, 16), (128, 64, 16), (256, 192, 16), (64, 32, 16), (64, 48, 0)]:
        tokens, hidden = 4096, 256
        gate_logits = torch.randn(tokens, n_experts) + torch.randn(1, n_experts) * 2
        # experts in families so that the groups are not ties
        base = torch.randn(n_experts // 8, hidden)
        expert_logits = base.repeat_interleave(8, 0) + 0.3 * torch.randn(n_experts, hidden)
        saliency = np.random.default_rng(n_experts + k).random(n_experts)

        orig_labels, orig_centres = orig_pseudo_group(
            saliency, expert_logits[:, None, :], k, gate_logits=gate_logits, group_size=c)
        sim_gate = dist2sim(gate_distance_from_gram(gate_logits.double().T @ gate_logits.double()))
        labels, centres = pseudo_group(saliency, expert_logits, k, sim_gate, c)
        assert (orig_centres == centres).all()
        same = (orig_labels == labels).mean()
        assert same == 1.0, f"E={n_experts} k={k} C={c}: {same:.1%} of labels agree"
        print(f"grouping parity E={n_experts} k={k} C={c}: identical "
              f"(group sizes {dict(zip(*np.unique(np.bincount(labels.astype(int)), return_counts=True)))})")

    x = torch.randn(2048, 384)
    a, b = orig_pca_reduce(x, r=64), pca_reduce(x, 64)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()
    print("pca parity: identical")


@torch.no_grad()
def check_alignment():
    torch.manual_seed(1)
    h, d, n_tok = 64, 48, 512
    x = torch.randn(n_tok, h)
    for cls in (Gated, Relu2):
        centre, member = cls(h, d), cls(h, d)
        perm = torch.randperm(d)
        params = {f"{n}.weight": m.weight.detach().clone() for n, m in centre.named_children()}
        shuffled = permute_neurons(params, perm)
        for n, m in member.named_children():
            m.weight.data = shuffled[f"{n}.weight"]
        # the shuffled expert computes the same function
        assert torch.allclose(centre(x), member(x), atol=1e-5)

        def act(e):  # (D, N) neuron activations
            pre = e.up_proj(x)
            a = torch.nn.functional.silu(e.gate_proj(x)) * pre if cls is Gated else torch.relu(pre) ** 2
            return a.T

        w = pca_reduce(torch.cat([neuron_weight_matrix(centre), neuron_weight_matrix(member)]), 16)
        found = alignment_permutation(act(centre), act(member), w[:d], w[d:])
        # found maps centre neuron i -> member neuron found[i]; the member's neuron perm^-1[i] is centre's i
        assert (perm[torch.as_tensor(found)] == torch.arange(d)).all()
        merged = merge_group(
            [params, {f"{n}.weight": m.weight.detach() for n, m in member.named_children()}],
            np.array([0.3, 0.7]), [None, found])
        for key in params:
            assert torch.allclose(merged[key], params[key], atol=1e-5)
        print(f"alignment {cls.__name__}: shuffled copy recovered, merge returns the centre")


class ExpertsBox:  # stands in for LinearExperts2D: holds the activation of non-gated experts
    def __init__(self, act_fn):
        self.act_fn = act_fn


@torch.no_grad()
def check_dense_stats():
    """The batched all-expert pass equals running every expert on every token (the original's statistics).

    sum_t p_j(t) E_j(x_t) must equal down_j(sum_t p_j(t) a_j(t)), and the sampled activations must be the
    experts' own hidden activations of the sampled tokens, across batches and token chunks.
    """
    torch.manual_seed(2)
    h, d, n_exp = 32, 24, 6
    for cls in (Gated, Relu2):
        experts = [cls(h, d) for _ in range(n_exp)]
        if cls is Gated:
            for e in experts:
                e._apply_gate = lambda gu: torch.nn.functional.silu(gu.chunk(2, -1)[0]) * gu.chunk(2, -1)[1]
        else:
            for e in experts:
                e.act_fn = lambda u: torch.relu(u) ** 2
        box = ExpertsBox(lambda u: torch.relu(u) ** 2)
        mod = REAMMergingModifier(sparsity=0.5, chunk_tokens=37)
        stats = _LayerStats(n_exp, h, d, 64, torch.device("cpu"), "cpu", torch.float32, seed=0)
        ref_sum, n_tok, seen = torch.zeros(n_exp, h), 0, []
        for t in (50, 120, 90):
            x = torch.randn(t, h)
            logits = torch.randn(t, n_exp)
            stats.scores = logits.softmax(-1)
            stats.tokens += t
            stats.plan_reservoir(t)
            slots, slot_tokens = stats.slots.clone(), stats.slot_tokens.clone()
            mod._dense_stats(stats, box, experts, x)
            for j, e in enumerate(experts):  # reference: every expert on every token
                ref_sum[j] += (stats.scores[:, j:j + 1] * e(x)).sum(0)
                pre = e.up_proj(x)
                own = e._apply_gate(torch.cat([e.gate_proj(x), pre], -1)) if cls is Gated else e.act_fn(pre)
                assert torch.allclose(stats.act[j, slots], own[slot_tokens], atol=1e-5), "sampled activations differ"
            n_tok += t
        mine = torch.stack([e.down_proj.weight @ stats.gated_act_sum[j] for j, e in enumerate(experts)]) / n_tok
        assert torch.allclose(mine, ref_sum / n_tok, atol=1e-5), (mine - ref_sum / n_tok).abs().max()
        print(f"dense statistics {cls.__name__}: equal to running every expert on every token "
              f"(max diff {(mine - ref_sum / n_tok).abs().max():.1e}); sampled activations equal")


def check_reservoir():
    n_experts, cap = 4, 1000
    stats = _LayerStats(n_experts, 8, 1, cap, torch.device("cpu"), "cpu", torch.float32, seed=0)
    seen = 0
    for n in [300, 500, 700, 1200, 50, 2000]:
        stats.plan_reservoir(n)
        token_ids = torch.arange(seen, seen + n, dtype=torch.float32)[:, None]
        for e in range(n_experts):  # every expert writes the global token id as its activation
            stats.act[e, stats.slots] = token_ids[stats.slot_tokens]
        seen += n
    ids = stats.act[:, :, 0]
    assert stats.res_filled == cap
    assert (ids == ids[0]).all(), "experts hold different tokens"
    assert len(torch.unique(ids[0])) == cap, "a token was sampled twice"
    # roughly uniform over the stream: mean id close to the stream mean
    assert abs(ids[0].mean().item() / (seen - 1) - 0.5) < 0.05
    print(f"reservoir: {cap} of {seen} tokens, identical across experts, no repeats, "
          f"mean position {ids[0].mean().item() / (seen - 1):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ream-repo", default=None, help="clone of SamsungSAILMontreal/ream at 84a3030")
    args = p.parse_args()
    if args.ream_repo:
        check_grouping(args.ream_repo)
    else:
        print("SKIPPED grouping and PCA parity with the original code: pass --ream-repo")
    check_alignment()
    check_dense_stats()
    check_reservoir()
    print("all checks passed" if args.ream_repo else "checks 3 and 4 passed; checks 1 and 2 were skipped")
