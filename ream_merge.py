"""REAM (Router-weighted Expert Activation Merging) as an llm-compressor modifier.

Port of https://github.com/SamsungSAILMontreal/ream (commit 84a3030, MIT licence; the grouping, PCA and
neuron-alignment code below is adapted from its `ream/ream.py`, `ream/weight_utils.py` and `ream/merger.py`,
Copyright (c) 2026 Samsung Electronics Co., Ltd.). Paper: https://arxiv.org/abs/2604.04356

Why a port: the original code assumes Qwen3-style decoder layers (`layer.mlp`, gated experts, softmax router,
attention-only layers) and pins transformers 4.57. llm-compressor already linearizes the experts of qwen3_moe,
qwen3_5_moe, glm4_moe_lite and nemotron_h models and runs REAP layer by layer, so REAM is implemented on the
same machinery: the same calibration data, the same REAP saliency, the same router
pruning and the same checkpoint format. The only difference to REAPPruningModifier is what happens to the
non-retained experts: REAP deletes them, REAM folds each into a similar retained expert.

The published evaluation ran it end to end on Qwen3-30B-A3B-Instruct-2507 (qwen3_moe) and
NVIDIA-Nemotron-3.5-Lightning-30B-A3B (nemotron_h) with llm-compressor 0.15.0. The other two architectures go
through the same code paths and are not part of that evaluation.

Algorithm per MoE layer (REAM defaults: grouping="ream", merging="logits+weights", saliency="reap",
group_size C=16, use_gate_output, gated_sim, sequential):
  1. Calibration statistics. The model's forward pass runs only the routed experts, as for REAP
     (moe_calibrate_all_experts=False); what REAM needs about every expert on every token is computed in one
     batched pass per batch:
     - REAP saliency per expert (llm-compressor's REAPSaliencyTracker and the same expert hooks as
       REAPPruningModifier, so the saliency is REAP's);
     - the mean over tokens of each expert's output weighted by its dense router score ("expert logits"). The
       experts are linear after their activation, so sum_t p_j(t) E_j(x_t) = down_j(sum_t p_j(t) a_j(t)) + bias:
       only the gate/up projections of all experts run on all tokens (one batched matmul), and down_proj is
       applied once per expert at merge time;
     - the Gram matrix of router logits over tokens (for router-logit similarity between experts);
     - the hidden (pre-down_proj) activations a_j of every expert on one shared random sample of tokens.
  2. The k = (1 - sparsity) * E most salient experts become group centres. Expert distance is
     1 - (cosine similarity of expert logits + router-logit similarity) / 2. Starting with the most salient
     centre, each centre takes up to C - 1 of its nearest unassigned experts. So a group holds at most C experts
     including its centre, exactly as the original `pseudo_group` does (the REAM paper's text reads "up to C
     unassigned non-centroid experts").
  3. Each non-centre expert in a group is neuron-aligned to its centre (Hungarian matching on the distance
     between neuron activations plus the distance between PCA-reduced neuron weights), then the group is
     averaged with saliency weights into the centre.
  4. The non-centre experts are removed and the router is cut to the centres (llm-compressor's
     prune_moe_layer, as in REAP). The sequential pipeline then propagates the calibration activations through
     the merged layer, so later layers are scored on the merged model.

Report (report_path): per MoE layer the routing counts, REAP saliency, retained experts, groups (centre first) and
merge weights, in report.json; and the (E, E) expert similarities the grouping used (output, router, combined), in
similarity.npz next to it.

Deviations from the original code (all deliberate):
  - Dense router scores use the model's own scoring function: softmax for the Qwen routers, sigmoid for the
    DeepSeek-V3-style routers of GLM-4.7-Flash and Nemotron (detected by `e_score_correction_bias`). The
    original always applies softmax, which matches only the Qwen models.
  - Neuron activations are sampled on the same tokens for all experts of a layer (reservoir sampling over the
    whole calibration set). The original samples tokens independently for each expert, so the activation
    distances in its alignment cost compare different tokens.
  - The weighted average is accumulated in float32 and cast once; the original casts after every addition.
  - REAP saliency comes from the router's actual top-k weights (as in REAPPruningModifier), not from the
    un-renormalized top-k softmax probabilities that the original recomputes from logits. The two differ whenever
    the router renormalizes its top-k weights (Qwen3 with norm_topk_prob, Qwen3.6, GLM, Nemotron). Using the
    actual weights keeps REAM's centres equal to the experts REAPPruningModifier retains.
  - Group-limited routing with n_group > 1 and multi-GPU (distributed) calibration are not supported; none of
    the four architectures above uses group-limited routing (n_group is 1 or absent).
"""

import json
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from compressed_tensors import align_module_device
from compressed_tensors.distributed import is_distributed
from compressed_tensors.offload import get_execution_device, update_offload_parameter
from loguru import logger
from pydantic import Field, PrivateAttr, model_validator
from scipy.optimize import linear_sum_assignment

from llmcompressor.core import Event, State
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.reap.utils import (
    MoeModelAttrs,
    REAPSaliencyTracker,
    get_moe_attrs,
    prune_moe_layer,
    update_model_config,
)

__all__ = ["REAMMergingModifier", "pseudo_group", "pca_reduce", "neuron_weight_matrix", "merge_group"]

PCA_DIM = 64  # REAM's heuristic dimension of the PCA-reduced neuron weights


# ---------------------------------------------------------------------------
# REAM math (adapted from SamsungSAILMontreal/ream)
# ---------------------------------------------------------------------------


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.linalg.norm(x, dim=-1, keepdim=True) + eps)


def dist2sim(d: np.ndarray) -> np.ndarray:
    """Distance matrix -> similarity in [0, 1] (REAM's utils.dist2sim with remove_self_loops=False)."""
    return 1 - d / d.max()


def gate_distance_from_gram(gram: torch.Tensor) -> np.ndarray:
    """Pairwise Euclidean distance between the L2-normalized router-logit columns of the experts.

    REAM stores all router logits L (tokens x E) and computes cdist(normalize(L^T), normalize(L^T)); with the
    Gram matrix G = L^T L accumulated over tokens this is sqrt(2 - 2 G_ij / (|L_i| |L_j|)), without storing L.
    """
    gram = gram.double()
    norms = gram.diagonal().clamp_min(0).sqrt()
    cos = gram / (norms[:, None] * norms[None, :] + 1e-8)
    return (2 - 2 * cos).clamp_min(0).sqrt().float().cpu().numpy()


@torch.no_grad()
def output_similarity(expert_logits: torch.Tensor) -> np.ndarray:
    """(E, E) similarity in [0, 1] of the experts' mean outputs: (cosine + 1) / 2, as REAM's expert_similarity
    on (1, H) tensors."""
    x = expert_logits.float()
    return ((F.cosine_similarity(x[:, None, :], x[None, :, :], dim=-1) + 1) / 2).cpu().numpy()


@torch.no_grad()
def pseudo_group(
    saliency: np.ndarray,
    expert_logits: torch.Tensor,
    k: int,
    sim_gate: np.ndarray | None = None,
    group_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """REAM's pseudo-pruning grouping.

    :param saliency: (E,) expert saliency
    :param expert_logits: (E, H) mean router-score-weighted output of each expert
    :param k: number of groups (= experts kept)
    :param sim_gate: (E, E) router-logit similarity in [0, 1], or None
    :param group_size: REAM's C; 0 uses MC-SMoE-style nearest-centre assignment
    :return: cluster label per expert (E,), centre indices (k,) in decreasing saliency
    """
    n_experts = expert_logits.shape[0]
    sim_out = output_similarity(expert_logits)
    if sim_gate is not None:
        sim_out = (sim_out + sim_gate) / 2
    d = (1 - sim_out).astype(np.float32)
    np.fill_diagonal(d, 0)
    assert np.isfinite(d).all()

    centroid_inds = np.argsort(saliency)[::-1].copy()[:k]
    labels = np.zeros(n_experts) - 1
    labels[centroid_inds] = np.arange(k)

    if group_size > 0:
        if k < n_experts // group_size:
            raise ValueError(f"group_size={group_size} too small for k={k} of {n_experts} experts")
        for centroid in centroid_inds:  # most salient centres pick first
            group = [centroid]
            for ind in np.argsort(d[centroid]):
                if ind != centroid and labels[ind] == -1:
                    group.append(ind)
                if len(group) >= group_size:
                    break
            for ind in group:
                labels[ind] = labels[centroid]
            if (labels >= 0).all():
                break
    else:
        for j in range(n_experts):
            if labels[j] < 0:
                labels[j] = np.argmin(d[centroid_inds, j])

    assert (labels >= 0).all(), labels
    return labels, centroid_inds


def pca_reduce(x: torch.Tensor, r: int) -> torch.Tensor:
    """Project rows of x (N, F) on their top-r principal directions, scaled by the singular values."""
    mean = x.mean(dim=-2, keepdim=True)
    x = x - mean
    with torch.random.fork_rng(devices=[x.device] if x.is_cuda else []):
        torch.manual_seed(0)  # torch.pca_lowrank is randomized; REAM seeds it with 0
        _, s, v = torch.pca_lowrank(x, q=r)
    return x @ (v[:, :r] * s[:r].reshape(1, r))


def expert_linears(expert: ExpertMLP) -> dict[str, torch.nn.Linear]:
    names = ["gate_proj", "up_proj", "down_proj"] if hasattr(expert, "gate_proj") else ["up_proj", "down_proj"]
    return {name: getattr(expert, name) for name in names}


def neuron_weight_matrix(expert: ExpertMLP) -> torch.Tensor:
    """(D, F) matrix with one row per hidden neuron: [gate[i, :] | up[i, :] | down[:, i]] (no gate for relu2
    experts), float32 on the execution device."""
    rows = []
    for name, linear in expert_linears(expert).items():
        w = linear.weight.detach().float()
        rows.append(w.t() if name == "down_proj" else w)
    return torch.cat(rows, dim=1)


def permute_neurons(params: dict[str, torch.Tensor], perm: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reorder the hidden neurons of one expert: rows of gate/up (and their biases), columns of down."""
    out = {}
    for key, t in params.items():
        name, kind = key.rsplit(".", 1)
        if name == "down_proj":
            out[key] = t[:, perm] if kind == "weight" else t  # down bias lives in the output space
        else:
            out[key] = t[perm]
    return out


def expert_params(expert: ExpertMLP) -> dict[str, torch.Tensor]:
    params = {}
    for name, linear in expert_linears(expert).items():
        params[f"{name}.weight"] = linear.weight.detach().float()
        if linear.bias is not None:
            params[f"{name}.bias"] = linear.bias.detach().float()
    return params


def alignment_permutation(
    act_centre: torch.Tensor | None,
    act_member: torch.Tensor | None,
    w_centre: torch.Tensor | None,
    w_member: torch.Tensor | None,
) -> np.ndarray:
    """Permutation of the member's neurons that best matches the centre's neurons (REAM's logits+weights).

    act_*: (D, N) neuron activations over the sampled tokens; w_*: (D, PCA_DIM) reduced neuron weights.
    """
    cost = 0
    if act_centre is not None:
        cost = cost + torch.cdist(normalize_rows(act_centre), normalize_rows(act_member)).cpu().numpy()
    if w_centre is not None:
        cost = cost + torch.cdist(normalize_rows(w_centre), normalize_rows(w_member)).cpu().numpy()
    _, col = linear_sum_assignment(cost)
    return col


def merge_group(
    group_params: list[dict[str, torch.Tensor]],
    weights: np.ndarray,
    perms: list[np.ndarray | None],
) -> dict[str, torch.Tensor]:
    """Saliency-weighted average of the experts of one group; group_params[0] is the centre (perm None)."""
    aligned = [
        params if perm is None else permute_neurons(params, torch.as_tensor(perm, device=next(iter(params.values())).device))
        for params, perm in zip(group_params, perms)
    ]
    return {key: sum(p[key] * float(w) for p, w in zip(aligned, weights)) for key in group_params[0]}


# ---------------------------------------------------------------------------
# Router scoring
# ---------------------------------------------------------------------------


def uses_sigmoid_router(router: torch.nn.Module) -> bool:
    """DeepSeek-V3-style routers (GLM) score experts with a sigmoid and steer the selection with a per-expert
    `e_score_correction_bias`; Qwen routers use a softmax. The router hook checks the choice against the router's
    own top-k weights."""
    return getattr(router, "e_score_correction_bias", None) is not None


# ---------------------------------------------------------------------------
# Calibration statistics
# ---------------------------------------------------------------------------


class _LayerStats:
    """Per-layer accumulators for REAM, on the layer's execution device (activations on `act_device`)."""

    def __init__(self, num_experts, hidden, intermediate, act_samples, device, act_device, dtype, seed):
        self.device = device
        self.saliency = REAPSaliencyTracker(num_experts, device)
        self.norms: dict[int, torch.Tensor] = {}
        # sum over tokens of p_j(t) * a_j(t) (hidden activations) and of p_j(t); down_proj is applied at merge time
        self.gated_act_sum = torch.zeros(num_experts, intermediate, dtype=torch.float32, device=device)
        self.score_sum = torch.zeros(num_experts, dtype=torch.float32, device=device)
        self.tokens = 0
        # stacked gate/up weights of all experts, (E, D, H), built when the layer first runs, freed at merge
        self.w_gate: torch.Tensor | None = None
        self.w_up: torch.Tensor | None = None
        self.b_gate: torch.Tensor | None = None
        self.b_up: torch.Tensor | None = None
        self.checked = False
        self.router_checked = False
        self.gram = torch.zeros(num_experts, num_experts, dtype=torch.float64, device=device)
        # shared token reservoir for neuron activations, (E, N, D); allocated when the layer first runs, so
        # only the layer being calibrated holds one
        self.act: torch.Tensor | None = None
        self.act_shape = (num_experts, act_samples, intermediate)
        self.act_dtype, self.act_device = dtype, act_device
        self.res_keys = torch.empty(0, device=device)
        self.res_filled = 0
        self.capacity = act_samples
        self.generator = torch.Generator(device=device).manual_seed(seed)
        # per-batch state set by the router hook and read by the expert hooks
        self.scores: torch.Tensor | None = None  # (T, E) dense router scores
        self.slots: torch.Tensor | None = None  # reservoir slots that get new tokens
        self.slot_tokens: torch.Tensor | None = None  # batch token index for each slot

    def plan_reservoir(self, num_tokens: int):
        """Choose which tokens of this batch enter the shared reservoir and which slots they take.

        Uniform sampling without replacement over all calibration tokens: every token gets a random key and
        the reservoir keeps the `capacity` largest keys seen so far.
        """
        if self.act is None:
            self.act = torch.zeros(self.act_shape, dtype=self.act_dtype, device=self.act_device)
        keys = torch.rand(num_tokens, generator=self.generator, device=self.device)
        all_keys = torch.cat([self.res_keys, keys])
        keep = torch.topk(all_keys, min(self.capacity, len(all_keys))).indices
        keep_old = keep[keep < self.res_filled]
        keep_new = keep[keep >= self.res_filled] - self.res_filled
        free = torch.ones(self.capacity, dtype=torch.bool, device=self.device)
        free[keep_old] = False
        slots = free.nonzero().squeeze(1)[: len(keep_new)]
        new_keys = torch.full((self.capacity,), -1.0, device=self.device)
        new_keys[keep_old] = self.res_keys[keep_old]
        new_keys[slots] = keys[keep_new]
        self.res_filled = min(self.capacity, self.res_filled + num_tokens)
        self.res_keys = new_keys[: self.res_filled]
        self.slots, self.slot_tokens = slots, keep_new


# ---------------------------------------------------------------------------
# Modifier
# ---------------------------------------------------------------------------


class REAMMergingModifier(Modifier):
    """Merge experts of MoE layers with REAM (see the module docstring).

    :param sparsity: fraction of experts removed per layer, in (0, 1); the same meaning as in
        REAPPruningModifier, so both methods give the same expert count.
    :param group_size: REAM's C, the maximum number of experts merged into one; 0 = MC-SMoE grouping.
    :param use_gate_output: add router-logit similarity to the expert distance.
    :param merging: "logits+weights" (REAM), "logits", "weights", or "avg_freq" (no neuron alignment).
    :param act_samples: calibration tokens sampled for the neuron activations (REAM: 1024 * 32).
    :param act_device: where the sampled activations are kept; None = the layer's execution device.
    :param chunk_tokens: tokens per chunk of the batched all-expert gate/up pass (bounds its memory).
    :param seed: seed of the token sampling.
    :param ignore: module name patterns to skip during MoE layer detection.
    :param report_path: optional .json report with saliency, counts and the groups of every layer.
    """

    requires_calibration_data: bool = True

    sparsity: float
    group_size: int = 16
    use_gate_output: bool = True
    merging: str = "logits+weights"
    act_samples: int = 1024 * 32
    act_device: Optional[str] = None
    chunk_tokens: int = 4096
    seed: int = 42
    ignore: list[str] = Field(default_factory=list)
    report_path: Optional[str] = Field(default=None)

    _moe_attrs: MoeModelAttrs | None = PrivateAttr(default=None)
    _n_keep: int = PrivateAttr(default=0)
    _stats: dict[str, _LayerStats] = PrivateAttr(default_factory=dict)
    _report: dict = PrivateAttr(default_factory=lambda: defaultdict(dict))
    _similarity: dict[str, dict[str, np.ndarray]] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "REAMMergingModifier":
        if not 0.0 < self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in (0, 1), got {self.sparsity}")
        if self.merging not in ("logits+weights", "logits", "weights", "avg_freq"):
            raise ValueError(f"unknown merging {self.merging!r}")
        if self.report_path is not None:
            if not self.report_path.endswith(".json"):
                raise ValueError(f"report_path must end with .json, got {self.report_path}")
            Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)
        return self

    def on_initialize(self, state: State, **kwargs) -> bool:
        if is_distributed():
            raise NotImplementedError("REAMMergingModifier supports single-process calibration only")
        self._moe_attrs = attrs = get_moe_attrs(state.model, self.ignore)
        if attrs.n_group is not None and attrs.n_group > 1:
            raise NotImplementedError(f"group-limited routing (n_group={attrs.n_group}) is not supported")
        self._n_keep = attrs.num_experts - int(attrs.num_experts * self.sparsity)
        if self._n_keep == attrs.num_experts:
            raise ValueError(f"sparsity={self.sparsity} removes no expert of {attrs.num_experts}")
        if attrs.top_k > self._n_keep:
            raise ValueError(f"top_k={attrs.top_k} > {self._n_keep} experts left; reduce the sparsity")
        if self.group_size > 0 and self._n_keep < attrs.num_experts // self.group_size:
            raise ValueError(f"group_size={self.group_size} too small to merge {attrs.num_experts} into {self._n_keep}")
        logger.info(
            f"REAM initialized: {len(attrs.moe_layer_names)} MoE layers, {attrs.num_experts} experts/layer, "
            f"merging into {self._n_keep} ({self.sparsity:.0%} removed), group_size={self.group_size}"
        )
        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        if get_calibrate_all_experts_flag():
            logger.info("REAM computes its all-expert statistics itself; moe_calibrate_all_experts=True only "
                        "makes calibration slower. Run oneshot with moe_calibrate_all_experts=False.")
        model, attrs = state.model, self._moe_attrs
        for layer_name in attrs.moe_layer_names:
            block = model.get_submodule(layer_name)
            experts = getattr(block, attrs.experts_attr)
            router = getattr(block, attrs.router_attr)
            expert_list = [e for e in experts.children() if isinstance(e, ExpertMLP)]
            first = expert_list[0]
            with align_module_device(first):
                hidden = first.down_proj.weight.shape[0]
                intermediate = first.down_proj.weight.shape[1]
                dtype = first.down_proj.weight.dtype
            device = get_execution_device(experts)
            self._stats[layer_name] = _LayerStats(
                attrs.num_experts, hidden, intermediate, self.act_samples,
                device, self.act_device or device, dtype, self.seed,
            )
            sigmoid = uses_sigmoid_router(router)
            self.register_hook(router, partial(self._router_hook, layer_name, sigmoid), "forward")
            # REAP's own per-expert hooks: output norms of the routed tokens, for the REAP saliency
            for idx, expert in enumerate(expert_list):
                self.register_hook(expert, partial(self._expert_hook, layer_name, idx), "forward")
            self.register_hook(experts, partial(self._experts_block_hook, layer_name, expert_list), "forward")

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self.remove_hooks()

    # -- hooks ---------------------------------------------------------------

    def _router_hook(self, layer_name, sigmoid, module, args, output):
        stats = self._stats.get(layer_name)
        if stats is None:
            return
        logits = (output[0] if isinstance(output, tuple) else output).detach().float()
        logits = logits.reshape(-1, logits.shape[-1])
        stats.scores = logits.sigmoid() if sigmoid else logits.softmax(dim=-1)
        if not stats.router_checked and isinstance(output, tuple) and len(output) == 3:
            # the chosen scoring must reproduce the router's own top-k weights up to their normalization
            weights, indices = output[1].detach().float(), output[2].long()
            mine = stats.scores.gather(1, indices.reshape(len(logits), -1))
            mine = mine / mine.sum(-1, keepdim=True)
            theirs = weights.reshape(len(logits), -1)
            theirs = theirs / theirs.sum(-1, keepdim=True)
            assert torch.allclose(mine, theirs, atol=2e-2), (
                f"{layer_name}: {'sigmoid' if sigmoid else 'softmax'} router scores do not reproduce the router's "
                f"top-k weights (max diff {(mine - theirs).abs().max():.3f})")
            stats.router_checked = True
        stats.gram += (logits.T @ logits).double()
        stats.tokens += logits.shape[0]
        stats.plan_reservoir(logits.shape[0])

    def _expert_hook(self, layer_name, idx, module, args, output):
        """As REAPPruningModifier: per-token output norms of the tokens routed to this expert."""
        stats = self._stats.get(layer_name)
        if stats is None:
            return
        out = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            stats.norms[idx] = torch.linalg.norm(out.float(), dim=-1).reshape(-1)

    @torch.no_grad()
    def _experts_block_hook(self, layer_name, expert_list, module, args, output):
        stats = self._stats.get(layer_name)
        if stats is None:
            return
        stats.saliency.update(args[1], args[2], stats.norms)
        stats.norms = {}
        self._dense_stats(stats, module, expert_list, args[0].detach())

    @torch.no_grad()
    def _dense_stats(self, stats: _LayerStats, experts, expert_list, x: torch.Tensor):
        """Hidden activations of every expert on every token of the batch, in one batched pass per chunk.

        Accumulates sum_t p_j(t) a_j(t) and writes the reservoir tokens' activations.
        """
        gated = hasattr(expert_list[0], "gate_proj")
        if stats.w_up is None:
            stats.w_up = torch.stack([e.up_proj.weight.detach() for e in expert_list])  # (E, D, H)
            if expert_list[0].up_proj.bias is not None:
                stats.b_up = torch.stack([e.up_proj.bias.detach() for e in expert_list])[:, None, :]
            if gated:
                stats.w_gate = torch.stack([e.gate_proj.weight.detach() for e in expert_list])
                if expert_list[0].gate_proj.bias is not None:
                    stats.b_gate = torch.stack([e.gate_proj.bias.detach() for e in expert_list])[:, None, :]

        def activations(xc):  # (T, H) -> (E, T, D), with the experts' own activation / gating function
            up = torch.matmul(xc, stats.w_up.transpose(1, 2))
            if stats.b_up is not None:
                up = up + stats.b_up
            if not gated:
                return experts.act_fn(up)
            gate = torch.matmul(xc, stats.w_gate.transpose(1, 2))
            if stats.b_gate is not None:
                gate = gate + stats.b_gate
            return expert_list[0]._apply_gate(torch.cat([gate, up], dim=-1))

        if not stats.checked:  # the batched pass must reproduce the experts' own computation
            xs = x[:4]
            mine = activations(xs)
            for j in (0, len(expert_list) - 1):
                e = expert_list[j]
                own = e.act_fn(e.up_proj(xs)) if not gated else e._apply_gate(
                    torch.cat([e.gate_proj(xs), e.up_proj(xs)], dim=-1))
                assert torch.allclose(mine[j].float(), own.float(), rtol=2e-2, atol=2e-2), \
                    f"batched expert activations differ from expert {j}'s own: {(mine[j] - own).abs().max()}"
            stats.checked = True

        scores = stats.scores  # (T, E) dense router scores of this batch
        slots, slot_tokens = stats.slots, stats.slot_tokens
        for start in range(0, x.shape[0], self.chunk_tokens):
            end = min(start + self.chunk_tokens, x.shape[0])
            act = activations(x[start:end])  # (E, t, D)
            p = scores[start:end]  # (t, E)
            stats.gated_act_sum += torch.einsum("te,etd->ed", p, act.float())
            stats.score_sum += p.sum(0)
            if slots is not None and len(slots):
                in_chunk = (slot_tokens >= start) & (slot_tokens < end)
                if in_chunk.any():
                    rows = act[:, slot_tokens[in_chunk] - start]  # (E, n, D)
                    stats.act[:, slots[in_chunk].to(stats.act.device)] = rows.to(stats.act.device, stats.act.dtype)
            del act

    # -- merging -------------------------------------------------------------

    def on_sequential_epoch_end(self, state: State, event: Event, **kwargs):
        merged_layers = 0
        for layer_name in sorted(self._stats):
            stats = self._stats[layer_name]
            if stats.saliency.total_count == 0:
                continue  # this layer's subgraph has not run yet
            self._merge_layer(state.model, layer_name, stats)
            del self._stats[layer_name]
            merged_layers += 1
        if merged_layers:
            logger.info(f"REAM merged experts in {merged_layers} layer(s)")

    @torch.no_grad()
    def _merge_layer(self, model, layer_name, stats: _LayerStats):
        attrs = self._moe_attrs
        block = model.get_submodule(layer_name)
        experts = getattr(block, attrs.experts_attr)
        expert_list = [e for e in experts.children() if isinstance(e, ExpertMLP)]
        device = stats.device

        saliency = stats.saliency.mean_saliency.cpu().numpy().astype(np.float64)
        counts = stats.saliency.count.cpu().numpy()
        report = {"count": counts.tolist(), "saliency": saliency.tolist()}
        zeros = saliency == 0
        if zeros.any():  # never-routed experts get the smallest non-zero saliency so they can be merged
            assert not zeros.all(), "no expert was routed any token"
            saliency[zeros] = min(0.5, saliency[~zeros].min())

        # mean router-score-weighted expert output: down_j(sum_t p_j a_j) + b_j sum_t p_j, over all tokens
        stats.w_gate = stats.w_up = stats.b_gate = stats.b_up = None
        rows = []
        for j, expert in enumerate(expert_list):
            with align_module_device(expert):
                down = expert.down_proj
                out = down.weight.detach().float() @ stats.gated_act_sum[j].to(down.weight.device)
                if down.bias is not None:
                    out = out + down.bias.detach().float() * stats.score_sum[j].to(down.bias.device)
            rows.append(out.to(device))
        expert_logits = torch.stack(rows) / max(stats.tokens, 1)
        sim_gate = dist2sim(gate_distance_from_gram(stats.gram)) if self.use_gate_output else None
        labels, centres = pseudo_group(saliency, expert_logits, self._n_keep, sim_gate, self.group_size)
        # the expert similarities the grouping used, for the report's similarity.npz (original expert numbering)
        sim_out = output_similarity(expert_logits)
        self._similarity[layer_name] = {
            "output": sim_out.astype(np.float32),
            "router": (sim_gate if sim_gate is not None else np.full_like(sim_out, np.nan)).astype(np.float32),
            "combined": ((sim_out + sim_gate) / 2 if sim_gate is not None else sim_out).astype(np.float32),
        }

        groups = []
        for j in range(self._n_keep):
            group = [int(i) for i in np.where(labels == j)[0]]
            group.sort(key=lambda g: g not in centres)  # centre first
            groups.append(group)

        use_act = "logits" in self.merging
        use_weights = "weights" in self.merging
        w_reduced = None
        if use_weights and any(len(g) > 1 for g in groups):
            mats = []
            for expert in expert_list:
                with align_module_device(expert):
                    mats.append(neuron_weight_matrix(expert).to(device))
            mats = normalize_rows(torch.stack(mats))  # (E, D, F)
            n_exp, d_int, _ = mats.shape
            w_reduced = pca_reduce(mats.flatten(0, 1), PCA_DIM).reshape(n_exp, d_int, PCA_DIM)
            del mats
        n_act = stats.res_filled

        merge_weights = []
        for group in groups:
            w = saliency[group] / saliency[group].sum()
            merge_weights.append(w.tolist())
            if len(group) == 1:
                continue
            centre = group[0]
            act_c = stats.act[centre, :n_act].to(device).float().T if use_act else None
            params, perms = [], [None]
            with align_module_device(expert_list[centre]):
                params.append(expert_params(expert_list[centre]))
            for member in group[1:]:
                with align_module_device(expert_list[member]):
                    params.append({k: v.to(device) for k, v in expert_params(expert_list[member]).items()})
                if self.merging == "avg_freq":
                    perms.append(None)
                    continue
                act_m = stats.act[member, :n_act].to(device).float().T if use_act else None
                perms.append(alignment_permutation(
                    act_c, act_m,
                    w_reduced[centre] if use_weights else None,
                    w_reduced[member] if use_weights else None,
                ))
            params[0] = {k: v.to(device) for k, v in params[0].items()}
            merged = merge_group(params, w, perms)
            for name, linear in expert_linears(expert_list[centre]).items():
                for kind in ("weight", "bias"):
                    key = f"{name}.{kind}"
                    if key in merged:
                        old = getattr(linear, kind)
                        update_offload_parameter(linear, kind, merged[key].to(old.dtype))

        retained = sorted(int(c) for c in centres)
        prune_moe_layer(model, layer_name, retained, attrs)

        report.update({
            "retained": retained,
            "groups": groups,
            "merge_weights": merge_weights,
            "act_tokens": n_act,
            "tokens": stats.tokens,
        })
        self._report[layer_name] = report
        sizes = np.bincount([len(g) for g in groups])
        logger.info(f"REAM {layer_name}: {attrs.num_experts} -> {len(retained)} experts; "
                    f"group sizes {dict((i, int(n)) for i, n in enumerate(sizes) if n)}")

    def on_finalize(self, state: State, **kwargs) -> bool:
        if self._stats:
            logger.warning(f"REAM: {len(self._stats)} MoE layer(s) never saw calibration tokens and were not merged")
        update_model_config(state.model, self._moe_attrs, self._n_keep)
        if self.report_path is not None:
            names = self._moe_attrs.moe_layer_names
            report = {key: [self._report[n].get(key, []) for n in names]
                      for key in ("count", "saliency", "retained", "groups", "merge_weights")}
            report["layers"] = names
            report["config"] = {
                "sparsity": self.sparsity, "group_size": self.group_size, "use_gate_output": self.use_gate_output,
                "merging": self.merging, "act_samples": self.act_samples, "seed": self.seed,
                "act_tokens": [self._report[n].get("act_tokens") for n in names],
                "tokens": [self._report[n].get("tokens") for n in names],
            }
            report["config"]["similarity_file"] = "similarity.npz"
            Path(self.report_path).write_text(json.dumps(report))
            # per MoE layer, (E, E) expert similarities in [0, 1] (NaN for a layer that was not merged):
            #   output   - of the experts' mean router-weighted outputs
            #   router   - of the experts' router-logit columns (NaN without use_gate_output)
            #   combined - what the grouping used: (output + router) / 2; distance = 1 - combined
            n = self._moe_attrs.num_experts
            empty = np.full((n, n), np.nan, dtype=np.float32)
            np.savez_compressed(
                Path(self.report_path).with_name("similarity.npz"), layers=np.array(names),
                **{kind: np.stack([self._similarity.get(name, {}).get(kind, empty) for name in names])
                   for kind in ("output", "router", "combined")})
        self._stats.clear()
        return True
