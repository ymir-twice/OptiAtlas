"""Deterministic image framing for supervised training.

Center cropping is opt in: it removes synthetic padding without stretching
geometry, but can remove captioned objects. The area limit is a hard guard,
not a claim that everything inside it is semantically safe.
"""
from __future__ import annotations

import math
from numbers import Real

from PIL import Image, ImageOps


def preprocessing_geometry(source_size, target_size, *, resize_mode="letterbox", max_crop_fraction=0.1):
    """Return JSON-safe geometry; coordinates are in the oriented source image.

    ``crop_fraction`` is the fraction of source area removed, before resampling.
    This computation is also usable by metadata-only dataset preflight tools.
    """
    for name, size in (("source_size", source_size), ("target_size", target_size)):
        if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size):
            raise ValueError(f"{name} must contain positive integer width and height")
    if resize_mode not in {"letterbox", "stretch", "center_crop"}:
        raise ValueError("resize_mode must be letterbox, stretch or center_crop")
    if (isinstance(max_crop_fraction, bool) or not isinstance(max_crop_fraction, Real)
            or not math.isfinite(max_crop_fraction) or not 0 <= max_crop_fraction < 1):
        raise ValueError("max_crop_fraction must be finite and within [0, 1)")
    width, height = source_size
    out_width, out_height = target_size
    crop_width, crop_height = float(width), float(height)
    if resize_mode == "center_crop":
        source_ratio, output_ratio = width / height, out_width / out_height
        if source_ratio > output_ratio:
            crop_width = output_ratio * height
        elif source_ratio < output_ratio:
            crop_height = width / output_ratio
    fraction = max(0.0, 1.0 - (crop_width / width) * (crop_height / height))
    if fraction > float(max_crop_fraction) + 1e-12:
        raise ValueError(
            f"center_crop would remove {fraction:.6%} of source area, exceeding "
            f"max_crop_fraction={float(max_crop_fraction):.6%}; source={source_size}, target={target_size}"
        )
    left, top = (width - crop_width) / 2, (height - crop_height) / 2
    contained_width, contained_height = out_width, out_height
    if resize_mode == "letterbox":
        if width / height > out_width / out_height:
            contained_height = round(height / width * out_width)
        elif width / height < out_width / out_height:
            contained_width = round(width / height * out_height)
    pad_left = round((out_width - contained_width) / 2)
    pad_top = round((out_height - contained_height) / 2)
    return {
        "resize_mode": resize_mode,
        "source_size": [width, height],
        "target_size": [out_width, out_height],
        "source_crop_box": [left, top, left + crop_width, top + crop_height],
        "crop_fraction": fraction,
        "max_crop_fraction": float(max_crop_fraction),
        "adds_padding": (contained_width, contained_height) != (out_width, out_height),
        "output_content_box": [pad_left, pad_top, pad_left + contained_width, pad_top + contained_height],
    }


def prepare_training_image(source, *, width, height, color_mode, resize_mode="letterbox", max_crop_fraction=0.1):
    """Return the transformed RGB/RGBA PIL image and its framing geometry.

    Existing letterbox/stretch operations are deliberately unchanged. A crop
    uses one uniformly scaled floating-point source box rather than rounding
    each resized axis independently. No randomness is consumed.
    """
    if color_mode not in {"RGB", "RGBA"}:
        raise ValueError("color_mode must be RGB or RGBA")
    image = ImageOps.exif_transpose(source).convert(color_mode)
    geometry = preprocessing_geometry(image.size, (width, height), resize_mode=resize_mode,
                                      max_crop_fraction=max_crop_fraction)
    if resize_mode == "letterbox":
        color = (127, 127, 127) if color_mode == "RGB" else (127, 127, 127, 255)
        image = ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS, color=color)
    elif resize_mode == "stretch":
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    else:
        image = image.resize((width, height), Image.Resampling.LANCZOS, box=geometry["source_crop_box"])
    return image, geometry
