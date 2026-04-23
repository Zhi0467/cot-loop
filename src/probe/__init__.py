"""Loop probe package.

Keep package import lightweight so metadata-only utilities can import
`probe.<module>` without pulling in the torch-backed training surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "MODEL_ROLLOUT_DEFAULTS",
    "PROBE_DEFAULTS",
    "ProbeConfig",
    "RolloutConfig",
    "build_probe_model",
    "get_probe_config",
    "get_rollout_config",
    "probe_preset_choices",
    "ActivationDataset",
    "make_dataloader",
]

_CONFIG_EXPORTS = {
    "MODEL_ROLLOUT_DEFAULTS",
    "PROBE_DEFAULTS",
    "ProbeConfig",
    "RolloutConfig",
    "build_probe_model",
    "get_probe_config",
    "get_rollout_config",
    "probe_preset_choices",
}
_DATALOADER_EXPORTS = {
    "ActivationDataset",
    "make_dataloader",
}

if TYPE_CHECKING:
    from .configs import (
        MODEL_ROLLOUT_DEFAULTS,
        PROBE_DEFAULTS,
        ProbeConfig,
        RolloutConfig,
        build_probe_model,
        get_probe_config,
        get_rollout_config,
        probe_preset_choices,
    )
    from .dataloader import ActivationDataset, make_dataloader


def __getattr__(name: str):
    if name in _CONFIG_EXPORTS:
        from . import configs

        return getattr(configs, name)
    if name in _DATALOADER_EXPORTS:
        from . import dataloader

        return getattr(dataloader, name)
    raise AttributeError(f"module 'probe' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
