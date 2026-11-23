"""Rewrite a checkpoint's fused MoE experts into the per-expert layout of the original Hub checkpoints.

transformers 5 holds the experts of a layer as two fused tensors, and llm-compressor saves them that way:
    <layer>.mlp.experts.gate_up_proj  (E, 2 * D, H)   gate = [e, :D], up = [e, D:]
    <layer>.mlp.experts.down_proj     (E, H, D)
    (experts without a gate, e.g. Nemotron's: <layer>.experts.up_proj (E, D, H) instead of gate_up_proj)
The original Qwen3 and GLM-4.7-Flash checkpoints store one Linear per expert:
    <layer>.mlp.experts.<e>.gate_proj.weight (D, H), .up_proj.weight (D, H), .down_proj.weight (H, D)
vLLM's GLM-4.7-Flash loader reads only the per-expert layout, so every compressed checkpoint is rewritten into it
(compress.py does this after saving). Both layouts load to identical weights in transformers.

    python unfuse_experts.py <checkpoint dir>
"""

import json
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

FUSED = re.compile(r"^(?P<prefix>.*\.experts)\.(?P<name>gate_up_proj|up_proj|down_proj)$")


def unfuse_checkpoint(path: Path) -> int:
    """Rewrite the shards of `path` in place; returns the number of fused tensors split."""
    path = Path(path)
    index_path = path / "model.safetensors.index.json"
    # a small checkpoint is one model.safetensors without an index
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    shards = sorted(set(index["weight_map"].values())) if index else ["model.safetensors"]
    new_map, split = {}, 0
    for shard in shards:
        tensors = {}
        with safe_open(path / shard, "pt") as f:
            metadata = f.metadata()
            for key in f.keys():
                t = f.get_tensor(key)
                m = FUSED.match(key)
                if m is None:
                    tensors[key] = t
                    continue
                prefix, name = m["prefix"], m["name"]
                split += 1
                if name == "gate_up_proj":
                    d = t.shape[1] // 2
                    for e in range(t.shape[0]):
                        tensors[f"{prefix}.{e}.gate_proj.weight"] = t[e, :d].contiguous()
                        tensors[f"{prefix}.{e}.up_proj.weight"] = t[e, d:].contiguous()
                else:  # down_proj, or up_proj of experts without a gate (Nemotron's relu2 experts)
                    for e in range(t.shape[0]):
                        tensors[f"{prefix}.{e}.{name}.weight"] = t[e].contiguous()
        save_file(tensors, path / shard, metadata=metadata or {"format": "pt"})
        new_map.update({k: shard for k in tensors})
    if index:
        index["weight_map"] = dict(sorted(new_map.items()))
        index_path.write_text(json.dumps(index, indent=2))
    return split


if __name__ == "__main__":
    n = unfuse_checkpoint(Path(sys.argv[1]).expanduser())
    print(f"split {n} fused expert tensors")
