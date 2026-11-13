"""Reward-only row identity, kept separate from text conditioning and tensors.

Captions need not be unique. Stable trajectory indices, rather than prompt text,
select metadata after sampling, filtering, permutation and microbatch slicing.
This module does not load GT images or add them to generation conditioning.
"""
from collections.abc import Mapping
from copy import deepcopy
import json
from operator import index
from pathlib import Path


def snapshot_reward_batch(prompts, metadata):
    """Validate and detach row context from mutable dataset/caller containers."""
    if len(prompts) != len(metadata):
        raise ValueError("Reward prompts and metadata must have equal batch lengths.")
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise TypeError("Reward prompts must be strings.")
    if any(not isinstance(row, Mapping) for row in metadata):
        raise TypeError("Each reward metadata entry must be a mapping.")
    # Empty metadata remains valid for existing prompt-only rewards/datasets.
    # GT-dependent scorers independently require and validate their image field.
    return list(prompts), [deepcopy(dict(row)) for row in metadata]


def reference_tracker_keys(prompts, metadata):
    """Group paired rewards by source AND GT, never by caption alone.

    Keep visits to the same source across sampling batches in the same group,
    matching the existing per-outer-epoch tracker scope. Component-normalized
    rewards have a separate, stricter batch/K identity in ``group_rewards``.
    This helper is only for reference-image rewards; text-only upstream rewards
    continue to use complete captions as their group keys.
    """
    prompts, metadata = snapshot_reward_batch(prompts, metadata)
    keys = []
    for prompt, row in zip(prompts, metadata):
        source = row.get("id")
        if source is None or not str(source):
            raise ValueError("Reference reward grouping requires a stable source id")
        digest = row.get("reference_image_sha256")
        if digest:
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdefABCDEF" for c in digest)):
                raise ValueError("Reference reward grouping received an invalid image SHA256")
            reference = ["sha256", digest.lower()]
        else:
            image = row.get("image")
            if not isinstance(image, str) or not image:
                raise ValueError("Reference reward grouping requires an image path or SHA256")
            reference = ["path", str(Path(image).expanduser().resolve())]
        keys.append(json.dumps([str(source), prompt, reference], ensure_ascii=False))
    return keys


class RewardContext:
    """One immutable-in-practice context entry per sampled trajectory."""

    def __init__(self):
        self._entries = []

    def __len__(self):
        return len(self._entries)

    def append(self, prompts, metadata):
        prompts, metadata = snapshot_reward_batch(prompts, metadata)
        start = len(self)
        self._entries.extend(zip(prompts, metadata))
        return list(range(start, len(self)))

    def select(self, trajectory_indices):
        entries = []
        for value in trajectory_indices:
            if isinstance(value, bool):
                raise TypeError("Reward trajectory index must be an integer, not bool.")
            try:
                position = index(value)
            except TypeError as exc:
                raise TypeError("Reward trajectory index must be an integer.") from exc
            if not 0 <= position < len(self):
                raise IndexError(f"Reward trajectory index {position} is out of range.")
            entries.append(self._entries[position])
        return snapshot_reward_batch([row[0] for row in entries], [row[1] for row in entries])
