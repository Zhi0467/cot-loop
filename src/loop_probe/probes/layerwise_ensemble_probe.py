from collections.abc import Callable

import torch
from torch import nn


class LayerwiseEnsembleProbe(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        probe_factory: Callable[[], nn.Module],
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("LayerwiseEnsembleProbe num_layers must be >= 1.")
        self.probes = nn.ModuleList([probe_factory() for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "LayerwiseEnsembleProbe expects [batch, layer, hidden] inputs, "
                f"got shape {tuple(x.shape)}"
            )
        if x.size(1) != len(self.probes):
            raise ValueError(
                "Layer count mismatch between features and ensemble probes: "
                f"{x.size(1)} vs {len(self.probes)}"
            )
        logits = [probe(x[:, idx, :]) for idx, probe in enumerate(self.probes)]
        return torch.stack(logits, dim=1)
