"""Projected CLIP features across Transformers versions, without detaching."""

import torch


def projected_features(output):
    """Unwrap the projected [batch, embedding] tensor, preserving dtype/grad.

    Transformers 4 returns a Tensor. Transformers 5 returns a model output
    whose pooler_output has already passed through CLIP's projection layer.
    """
    features = output if torch.is_tensor(output) else getattr(output, "pooler_output", None)
    if not torch.is_tensor(features) or features.ndim != 2:
        raise TypeError("CLIP feature API must return projected [batch, embedding] features")
    return features
