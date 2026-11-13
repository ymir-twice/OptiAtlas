"""Model-independent training objectives; importing this package loads no models."""

from .flow_matching import flow_matching_loss, flow_matching_sample
from .opsd import (
    RewardTargets,
    behavior_ema_decay,
    build_reward_targets,
    group_normalized_advantages,
    opsd_loss,
)

__all__ = [
    "flow_matching_loss", "flow_matching_sample", "RewardTargets",
    "behavior_ema_decay", "build_reward_targets", "group_normalized_advantages", "opsd_loss",
]
