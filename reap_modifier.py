"""REAP pruning that also reports which experts each layer kept and removed.

llm-compressor's REAPPruningModifier writes only `count` and `saliency` per MoE layer to its report. This subclass
records the experts its own selection (`REAPSaliencyTracker.compute_retained_experts`) keeps, and adds to the report:
    retained  per layer, the kept expert ids (original numbering, ascending: the order of the pruned checkpoint)
    removed   per layer, the dropped expert ids
    layers    the MoE layer names, in the order of every per-layer list
    config    sparsity
so a REAP report has the same per-layer fields as a REAM report (ream_merge.py), minus groups and merge weights.
Pruning itself is unchanged.
"""

import json
from pathlib import Path

from llmcompressor.core import Event, State
from llmcompressor.modifiers.pruning import REAPPruningModifier


class REAPRecordingModifier(REAPPruningModifier):
    def on_sequential_epoch_end(self, state: State, event: Event, **kwargs):
        # wrap each pending layer's selection, so the experts REAP keeps are recorded as it picks them
        for layer_name, tracker in self._saliency_trackers.items():
            if "compute_retained_experts" in vars(tracker):
                continue  # already wrapped in an earlier epoch

            def record(*args, _select=tracker.compute_retained_experts, _layer=layer_name, **kw):
                retained = _select(*args, **kw)
                self._report[_layer]["retained"] = sorted(int(i) for i in retained.tolist())
                return retained

            tracker.compute_retained_experts = record
        return super().on_sequential_epoch_end(state, event, **kwargs)

    def on_finalize(self, state: State, **kwargs) -> bool:
        names = list(self._moe_attrs.moe_layer_names)
        num_experts = self._moe_attrs.num_experts
        result = super().on_finalize(state, **kwargs)  # writes count and saliency
        if self.report_path is not None:
            path = Path(self.report_path)
            report = json.loads(path.read_text())
            retained = [self._report[n].get("retained", list(range(num_experts))) for n in names]
            report["retained"] = retained
            report["removed"] = [sorted(set(range(num_experts)) - set(r)) for r in retained]
            report["layers"] = names
            report["config"] = {"sparsity": self.sparsity}
            path.write_text(json.dumps(report))
        return result
