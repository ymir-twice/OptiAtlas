"""Frozen DINOv2 GT cosine, negative RGB MSE, and HPSv2.1 prompt cosine."""
from __future__ import annotations

from .dinov3_hpsv2_mse import Dinov3HPSv2MSEReward


DINOV2_MODEL_NAME = "facebook/dinov2-base"
DINOV2_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"


class Dinov2HPSv2MSEReward(Dinov3HPSv2MSEReward):
    """Return raw components and their literal weighted sum.

    Prompt-group normalization is applied by the upstream reward wrapper using
    detached endpoint statistics. These raw differentiable components remain
    available for that normalization and for independent diagnostics.
    """

    backbone_family = "dinov2"
    dino_component_name = "dino_dense"
    reward_kind = "dinov2_hpsv2_mse"

    def __init__(self, model_name=DINOV2_MODEL_NAME, *, revision=DINOV2_REVISION,
                 dino_weight=1 / 3, mse_weight=1 / 3, hpsv2_weight=1 / 3,
                 hpsv2_version="v2.1", **kwargs):
        super().__init__(model_name=model_name, revision=revision,
                         dino_weight=dino_weight, mse_weight=mse_weight,
                         hpsv2_weight=hpsv2_weight, hpsv2_version=hpsv2_version,
                         **kwargs)
