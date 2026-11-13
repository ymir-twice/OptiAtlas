"""Reference reward for brightness and its regional spatial arrangement.

This explicitly defined pixel objective is not a learned physical score.
It does not construct or load any feature encoder.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .spatial import validate_reference_pair


class RegionalLuminanceReward:
    requires_reference_images = True
    supports_valid_mask = True

    def __init__(self, *, grids=(1, 2, 4, 8)):
        if (not grids or any(isinstance(g, bool) or not isinstance(g, int) or g < 1 for g in grids)
                or len(set(grids)) != len(grids)):
            raise ValueError("grids must be nonempty, distinct positive integers")
        self.grids = tuple(grids)
        self.metadata = {
            "kind": "regional_luminance", "grids": list(self.grids),
            "luminance": "Y = 0.2126 R + 0.7152 G + 0.0722 B on RGB [0,1]; "
                         "BT.709 luma coefficients, no sRGB gamma linearization",
            "reward": "-mean_over_grids(coverage_weighted_mean_abs(masked_region_mean_g(Ygen) - "
                      "masked_region_mean_g(Ygt)))",
            "region_definition": "full-frame adaptive average pooling to g by g spatial cells; "
                                 "optional valid_mask excludes upstream reference padding",
            "scale_weights": "equal per grid; cells weighted by valid-pixel fraction (equal if unpadded)",
            "components": ["regional_luminance"], "minimum": -1.0, "maximum": 0.0,
            "image_preprocessing": "original aligned full-frame pixels; no crop, resize or augmentation",
            "model": None, "model_dtype": "float32", "normalization": "none; independent per image",
            "reference_images": "required explicit aligned GT, detached; never input conditioning",
        }

    def reward_components(self, images, prompts, reference_images=None, valid_mask=None):
        validate_reference_pair(images, prompts, reference_images, valid_mask=valid_mask)
        if max(self.grids) > min(images.shape[-2:]):
            raise ValueError("regional luminance grids cannot exceed the input height or width")
        with torch.autocast(device_type=images.device.type, enabled=False):
            generated = images.float()
            reference = reference_images.detach().float()
            weights = generated.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
            generated_y = (generated * weights).sum(1, keepdim=True)
            reference_y = (reference * weights).sum(1, keepdim=True)
            mask = (torch.ones_like(generated_y) if valid_mask is None else valid_mask.detach().float())
            distances = []
            for grid in self.grids:
                coverage = F.adaptive_avg_pool2d(mask, (grid, grid))
                generated_cells = F.adaptive_avg_pool2d(generated_y * mask, (grid, grid)) / coverage.clamp_min(1e-8)
                target_cells = F.adaptive_avg_pool2d(reference_y * mask, (grid, grid)) / coverage.clamp_min(1e-8)
                distances.append(((generated_cells - target_cells).abs() * coverage).sum((1, 2, 3))
                                 / coverage.sum((1, 2, 3)).clamp_min(1e-8))
            score = -torch.stack(distances).mean(0)
        return {"regional_luminance": score, "score": score}

    def __call__(self, images, prompts, reference_images=None, valid_mask=None):
        return self.reward_components(images, prompts, reference_images, valid_mask=valid_mask)["score"]
