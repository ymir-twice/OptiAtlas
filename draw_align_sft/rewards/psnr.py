"""Per-image differentiable PSNR against an explicit, detached RGB target.

Pixels are in [0, 1], hence MAX_I=1. An explicit MSE floor gives a finite
score at an exact match; all other scores are ordinary PSNR in decibels.
No CLIP/text score, learned weights, or batch-wise normalization is involved.
"""

import math

import torch

from .pixel import reference_mse


class PSNRReward:
    requires_reference_images = True

    def __init__(self, *, mse_floor=1e-8):
        if isinstance(mse_floor, bool) or not math.isfinite(float(mse_floor)) or not 0 < float(mse_floor) < 1:
            raise ValueError("PSNR mse_floor must be finite and strictly between zero and one")
        self.mse_floor = float(mse_floor)
        self.metadata = {"kind": "psnr", "data_range": 1.0, "unit": "dB",
                         "reduction": "mean over RGB and pixels, separately for each image",
                         "mse_floor": self.mse_floor, "maximum_db": -10 * math.log10(self.mse_floor),
                         "reference_images": "required, detached; never generation conditioning"}

    def __call__(self, images, prompts, reference_images=None):
        mse = reference_mse(images, prompts, reference_images)
        return -10.0 * torch.log10(mse.clamp_min(self.mse_floor))
