"""Literal DINOv3 GT cosine + HPSv2 prompt cosine - RGB MSE reward."""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from pathlib import Path

from .dino_dense import DinoDenseReward, DINOV3_MODEL_NAME, DINOV3_REVISION
from .mse import MSEReward


class Dinov3HPSv2MSEReward:
    requires_reference_images = True
    supports_valid_mask = True
    backbone_family = "dinov3"
    dino_component_name = "dinov3_dense"
    reward_kind = "dinov3_hpsv2_mse"

    def __init__(self, model_name=DINOV3_MODEL_NAME, *, revision=DINOV3_REVISION,
                 device="cpu", image_size=336, microbatch_size=1, local_files_only=True,
                 dino_weight=1.0, hpsv2_weight=0.3, mse_weight=0.3,
                 hpsv2_model_path=None, hpsv2_microbatch_size=1, hpsv2_dtype="float32",
                 hpsv2_gradient_checkpointing=True, hpsv2_version="v2.0",
                 dino_scorer=None, hpsv2_scorer=None):
        from .hpsv2 import HPSv2Reward, hpsv2_checkpoint_spec
        hps_spec = hpsv2_checkpoint_spec(hpsv2_version)
        dino_component = self.dino_component_name
        family_label = "DINOv3" if self.backbone_family == "dinov3" else "DINOv2"
        weights = {dino_component: dino_weight, "hpsv2": hpsv2_weight, "negative_mse": mse_weight}
        for name, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} weight must be finite and nonnegative")
        if not any(weights.values()):
            raise ValueError("At least one reward weight must be positive")
        if hpsv2_scorer is None and not hpsv2_model_path:
            raise ValueError(f"hpsv2_model_path must point to the prepared local HPSv2 {hpsv2_version} directory")
        self.component_weights = {name: float(value) for name, value in weights.items()}
        self.dino = dino_scorer if dino_scorer is not None else DinoDenseReward(
            model_name=model_name, revision=revision, device=device, image_size=image_size,
            microbatch_size=microbatch_size, local_files_only=local_files_only,
            backbone_family=self.backbone_family,
        )
        self.mse = MSEReward()
        if hpsv2_scorer is None:
            hpsv2_scorer = HPSv2Reward(
                checkpoint_path=Path(hpsv2_model_path).expanduser() / hps_spec["filename"],
                device=device, dtype=hpsv2_dtype, microbatch_size=hpsv2_microbatch_size,
                gradient_checkpointing=hpsv2_gradient_checkpointing,
                hps_version=hpsv2_version,
            )
        declared_version = getattr(hpsv2_scorer, "metadata", {}).get("version")
        if declared_version is not None and declared_version != hpsv2_version:
            raise ValueError(f"Injected HPS scorer version {declared_version} differs from requested {hpsv2_version}")
        self.hpsv2 = hpsv2_scorer
        self.metadata = {
            "kind": self.reward_kind, "backbone_family": f"{family_label} + HPSv2 {hpsv2_version}",
            "hpsv2_version": hpsv2_version,
            "reward": f"dino_weight * {dino_component} + hpsv2_weight * hpsv2 + mse_weight * negative_mse",
            "component_weights": dict(self.component_weights), "components": list(self.component_weights),
            "weighted_components": [f"weighted_{name}" for name in self.component_weights],
            "normalization": "none; literal weighted sum; no component or gradient normalization",
            "reference_images": f"{family_label} and MSE compare the same aligned detached GT",
            "hpsv2_conditioning": "generated full canvas and corresponding original prompt; no GT input",
            "valid_mask": "GT valid mask weights DINO patches and RGB MSE; HPSv2 scores the full generated canvas",
            "minimum": -sum(self.component_weights.values()), "maximum": float(dino_weight + hpsv2_weight),
            "component_metadata": {
                dino_component: deepcopy(getattr(self.dino, "metadata", {})),
                "hpsv2": deepcopy(getattr(self.hpsv2, "metadata", {})),
                "negative_mse": deepcopy(self.mse.metadata),
            },
        }

    def reward_components(self, images, prompts, reference_images=None, valid_mask=None):
        components = {
            self.dino_component_name: self.dino(images, prompts, reference_images=reference_images, valid_mask=valid_mask),
            "hpsv2": self.hpsv2(images, prompts),
            "negative_mse": self.mse(images, prompts, reference_images=reference_images, valid_mask=valid_mask),
        }
        weighted = {f"weighted_{name}": self.component_weights[name] * value for name, value in components.items()}
        return {**components, **weighted, "score": sum(weighted.values())}

    def __call__(self, images, prompts, reference_images=None, valid_mask=None):
        return self.reward_components(images, prompts, reference_images, valid_mask)["score"]
