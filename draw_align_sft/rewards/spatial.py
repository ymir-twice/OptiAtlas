"""Differentiable, full-frame preprocessing shared by dense reference rewards.

Letterboxing retains the source aspect ratio up to integer pixel rounding.
Coverage weights exclude padding from the spatial score reduction. They do
not constitute an encoder attention mask: a boundary patch can contain both
real and padded pixels, and self-attention can still see padded tokens.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def validate_reference_pair(images, prompts, reference_images, valid_mask=None):
    """Validate aligned floating RGB [0,1] images without detaching generation."""
    if (not torch.is_tensor(images) or images.ndim != 4 or images.shape[1] != 3
            or images.shape[0] < 1 or min(images.shape[-2:]) < 1):
        raise ValueError("reward images must be nonempty [batch,3,height,width]")
    if not images.is_floating_point() or len(prompts) != images.shape[0]:
        raise ValueError("reward expects floating images aligned with prompts")
    if not torch.is_tensor(reference_images) or reference_images.shape != images.shape:
        raise ValueError("reward requires explicit reference_images with the same BCHW shape")
    if not reference_images.is_floating_point():
        raise ValueError("reference_images must be floating RGB pixels")
    if images.device != reference_images.device:
        raise ValueError("generated and reference images must be on the same device")
    for label, value in (("generated", images), ("reference", reference_images)):
        if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError(f"reward {label} pixels must be finite and in [0,1]")
    validate_valid_mask(images, valid_mask)


def validate_valid_mask(images, valid_mask):
    """Check optional upstream reference validity, never an optimization input."""
    if valid_mask is None:
        return
    expected_shape = (images.shape[0], 1, *images.shape[-2:])
    if not torch.is_tensor(valid_mask) or valid_mask.shape != expected_shape:
        raise ValueError("valid_mask must have shape [B,1,H,W] aligned with images")
    if valid_mask.device != images.device:
        raise ValueError("valid_mask and images must be on the same device")
    if (not torch.isfinite(valid_mask).all()
            or ((valid_mask < 0) | (valid_mask > 1)).any()):
        raise ValueError("valid_mask values must be finite and in [0,1]")
    if (valid_mask.sum((1, 2, 3)) <= 0).any():
        raise ValueError("valid_mask must include valid pixels in every image")


def letterbox_rgb(images, *, image_size: int, patch_size: int, valid_mask=None):
    """Return FP32 RGB square canvas and per-patch valid-pixel coverage [B,N].

    The long edge is resized to ``image_size``; the other edge is rounded to
    the nearest whole pixel, with at least one pixel retained. Padding is
    centered and filled with ImageNet mean (zero after normalization).
    Both images of a pair must have the same original spatial shape.
    """
    for name, value in (("image_size", image_size), ("patch_size", patch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if image_size % patch_size:
        raise ValueError("image_size must be divisible by patch_size")
    if (not torch.is_tensor(images) or images.ndim != 4 or images.shape[1] != 3
            or images.shape[0] < 1 or min(images.shape[-2:]) < 1
            or not images.is_floating_point()):
        raise ValueError("letterbox images must be nonempty floating [B,3,H,W]")
    validate_valid_mask(images, valid_mask)
    height, width = images.shape[-2:]
    scale = image_size / max(height, width)
    resized_height = max(1, min(image_size, int(round(height * scale))))
    resized_width = max(1, min(image_size, int(round(width * scale))))
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    padding = (left, image_size - resized_width - left,
               top, image_size - resized_height - top)
    pixels = F.interpolate(images.float(), size=(resized_height, resized_width),
                           mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    mean = pixels.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    # Padding a centered image avoids in-place assignment and keeps the
    # generated-image autograd path intact on CPU and CUDA.
    pixels = F.pad(pixels - mean, padding) + mean
    if valid_mask is None:
        resized_mask = pixels.new_ones((images.shape[0], 1, resized_height, resized_width))
    else:
        resized_mask = F.interpolate(valid_mask.detach().float(),
                                     size=(resized_height, resized_width), mode="area")
    valid_pixels = F.pad(resized_mask, padding)
    coverage = F.avg_pool2d(valid_pixels, kernel_size=patch_size, stride=patch_size)
    coverage = coverage.flatten(1)
    return pixels, coverage


def imagenet_normalize(pixels):
    mean = pixels.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


def coverage_mean(values, coverage):
    """Mean over corresponding patch values [B,N], ignoring pure padding."""
    if values.shape != coverage.shape:
        raise ValueError("patch values and coverage must have the same [B,N] shape")
    return (values * coverage).sum(-1) / coverage.sum(-1).clamp_min(1e-8)
