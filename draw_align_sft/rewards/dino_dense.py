"""Pure frozen DINOv2 or DINOv3 patch cosine reward against an aligned GT image."""
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from .spatial import (IMAGENET_MEAN, IMAGENET_STD, coverage_mean, imagenet_normalize,
                      letterbox_rgb, validate_reference_pair)


DINOV3_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"


class DinoDenseReward:
    """Only a spatial feature reward; no luminance/pixel objective is computed."""

    requires_reference_images = True
    supports_valid_mask = True

    def __init__(self, model_name=None, *, revision=None,
                 device="cpu", image_size=336, microbatch_size=2,
                 local_files_only=False, model=None, backbone_family="dinov2"):
        if backbone_family not in ("dinov2", "dinov3"):
            raise ValueError("backbone_family must be 'dinov2' or 'dinov3'")
        is_v3 = backbone_family == "dinov3"
        family_label = "DINOv3" if is_v3 else "DINOv2"
        component_name = "dinov3_dense" if is_v3 else "dino_dense"
        if model_name is None:
            model_name = DINOV3_MODEL_NAME if is_v3 else "facebook/dinov2-base"
        if revision is None and is_v3:
            revision = DINOV3_REVISION
        for name, value in (("image_size", image_size), ("microbatch_size", microbatch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        injected = model is not None
        if model is None:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name, revision=revision,
                                             local_files_only=local_files_only,
                                             torch_dtype=torch.float32)
        config = model.config
        model_type = getattr(config, "model_type", None)
        allowed_types = {"dinov3_vit"} if is_v3 else {"dinov2", "dinov2_with_registers"}
        if model_type not in allowed_types:
            raise ValueError(f"pure {family_label} reward requires a {family_label} ViT; "
                             f"got {model_type!r}")
        patch_size = getattr(config, "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
                raise ValueError("DINO config.patch_size must describe square patches")
            patch_size = patch_size[0]
        if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
            raise ValueError("DINO config.patch_size must be a positive integer")
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by the DINO patch_size")
        registers = getattr(config, "num_register_tokens", 0)
        if isinstance(registers, bool) or not isinstance(registers, int) or registers < 0:
            raise ValueError("num_register_tokens must be a nonnegative integer")
        self.device = torch.device(device)
        self.backbone_family = backbone_family
        self.component_name = component_name
        self.model = model.to(device=self.device, dtype=torch.float32).requires_grad_(False).eval()
        self.patch_size = patch_size
        self.num_register_tokens = registers
        self.image_size = image_size
        self.microbatch_size = microbatch_size
        self.metadata = {
            "kind": component_name, "backbone_family": family_label, "model_name": str(model_name),
            "revision": revision, "resolved_revision": getattr(config, "_commit_hash", None),
            "model_config_source": getattr(config, "_name_or_path", None),
            "injected_model": injected, "model_type": model_type,
            "model_dtype": "float32", "model_frozen": True,
            "local_files_only": bool(local_files_only), "microbatch_size": microbatch_size,
            "image_size": image_size, "patch_size": patch_size,
            "num_patches": (image_size // patch_size) ** 2, "num_register_tokens": registers,
            "feature_source": "final normalized last_hidden_state; CLS and registers excluded",
            "reward": "coverage-weighted mean of corresponding patch cosine similarities",
            "components": [component_name], "normalization": "none; independent per image",
            "image_preprocessing": "full-frame aspect-preserving bicubic resize, antialias=True; "
                                   "centered ImageNet-mean letterbox, no crop/augmentation",
            "patch_coverage": "fraction of real image pixels; pure padding excluded, "
                              "boundary patches weighted by coverage; not an attention mask",
            "image_mean": list(IMAGENET_MEAN), "image_std": list(IMAGENET_STD),
            "reference_images": "required explicit aligned GT, detached; never input conditioning",
        }

    def _patch_features(self, images, valid_mask=None):
        pixels, coverage = letterbox_rgb(images, image_size=self.image_size,
                                        patch_size=self.patch_size, valid_mask=valid_mask)
        output = self.model(pixel_values=imagenet_normalize(pixels))
        hidden = (output.get("last_hidden_state") if isinstance(output, Mapping)
                  else getattr(output, "last_hidden_state", None))
        prefix = 1 + self.num_register_tokens
        if (not torch.is_tensor(hidden) or hidden.ndim != 3
                or hidden.shape[0] != images.shape[0]
                or hidden.shape[1] != prefix + coverage.shape[1] or hidden.shape[2] < 1):
            raise ValueError("DINO last_hidden_state must contain exactly CLS, configured registers, "
                             "and one token per spatial patch")
        return hidden[:, prefix:, :].float(), coverage

    def reward_components(self, images, prompts, reference_images=None, valid_mask=None):
        validate_reference_pair(images, prompts, reference_images, valid_mask=valid_mask)
        self.model.eval()
        scores = []
        with torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(0, images.shape[0], self.microbatch_size):
                generated = images[start:start + self.microbatch_size].to(self.device, torch.float32)
                reference = reference_images[start:start + self.microbatch_size].detach().to(
                    self.device, torch.float32)
                mask = (None if valid_mask is None else
                        valid_mask[start:start + self.microbatch_size].detach().to(self.device, torch.float32))
                with torch.no_grad():
                    target_features, _ = self._patch_features(reference, mask)
                generated_features, coverage = self._patch_features(generated, mask)
                similarity = F.cosine_similarity(generated_features, target_features, dim=-1, eps=1e-8)
                scores.append(coverage_mean(similarity, coverage))
        score = torch.cat(scores).to(images.device)
        return {self.component_name: score, "score": score}

    def __call__(self, images, prompts, reference_images=None, valid_mask=None):
        return self.reward_components(images, prompts, reference_images, valid_mask=valid_mask)["score"]
