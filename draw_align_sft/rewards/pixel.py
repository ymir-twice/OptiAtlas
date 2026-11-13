"""Shared, per-image pixel MSE for rewards against fixed GT images."""
import torch


def reference_mse(images, prompts, reference_images, *, valid_mask=None):
    if not torch.is_tensor(images) or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Pixel reward images must be [batch,3,height,width]")
    if not len(prompts) or len(prompts) != images.shape[0] or not images.is_floating_point():
        raise ValueError("Pixel reward expects a nonempty aligned floating image batch")
    if not torch.is_tensor(reference_images) or reference_images.shape != images.shape:
        raise ValueError("Pixel reward requires explicit reference_images with the same BCHW shape")
    if not reference_images.is_floating_point():
        raise ValueError("reference_images must be floating pixels in [0,1]")
    if images.device != reference_images.device:
        raise ValueError("Generated and reference images must be on the same device")
    for label, value in (("generated", images), ("reference", reference_images)):
        if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError(f"Pixel reward {label} pixels must be finite and in [0,1]")
    # Fixed targets never receive gradients. Compute in FP32 before squaring,
    # even when the generator and decoder use BF16.
    error = (images.float() - reference_images.detach().float()).square()
    if valid_mask is None:
        return error.mean(dim=(1, 2, 3))
    from .spatial import validate_reference_pair
    validate_reference_pair(images, prompts, reference_images, valid_mask=valid_mask)
    mask = valid_mask.detach().to(device=images.device, dtype=torch.float32)
    return (error * mask).sum(dim=(1, 2, 3)) / (3 * mask.sum(dim=(1, 2, 3)))
