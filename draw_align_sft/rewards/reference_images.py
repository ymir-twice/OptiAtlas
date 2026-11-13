"""Deterministic target pixels for explicitly paired image rewards.

The target is the metadata's image, never an image looked up by caption. All
cached tensors stay on CPU and are detached FP32 RGB in [0, 1]. Geometry is an
explicit part of the reward objective: stretching preserves the entire scene
but changes aspect ratio; cropping can remove objects; letterboxing supervises
white padding as well as the image. There is no automatic mode fallback.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps
import torch

from ..data import ManifestPathResolver
from ..image_preprocessing import preprocessing_geometry


def _file_identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class ReferenceImageLoader:
    """Load target RGB pixels at the generated image's exact height and width.

    Relative paths require ``image_root`` (normally the JSONL's parent).
    ``reference_image_sha256`` is checked whenever supplied; ``require_sha256``
    makes it mandatory. File stat changes invalidate cached content and trigger
    a new read/hash. Immutable source data is expected while a batch is used.

    Callers sharing one loader across threads must serialize calls, including
    access to ``last_metadata``. Returned tensors and metadata are independent
    copies; mutating them cannot change a later cached target.
    """

    def __init__(self, *, image_root=None, resize_mode="center_crop",
                 max_crop_fraction=0.06, cache_size=32, require_sha256=False):
        # Reuse the supervised preprocessing geometry validator, but use white
        # alpha compositing/padding for RGB reward appearance, not FM gray pads.
        preprocessing_geometry((1, 1), (1, 1), resize_mode=resize_mode,
                               max_crop_fraction=max_crop_fraction)
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        if not isinstance(require_sha256, bool):
            raise ValueError("require_sha256 must be a bool")
        self.image_root = Path(image_root).expanduser().resolve() if image_root is not None else None
        self.resize_mode = resize_mode
        self.max_crop_fraction = float(max_crop_fraction)
        self.cache_size = cache_size
        self.require_sha256 = require_sha256
        self._resolver = ManifestPathResolver(self.image_root or Path("/"))
        self._cache = OrderedDict()
        self._last_metadata = []

    @property
    def metadata(self):
        return {
            "kind": "explicit_metadata_reference_image",
            "image_root": str(self.image_root) if self.image_root is not None else None,
            "resize_mode": self.resize_mode,
            "max_crop_fraction": self.max_crop_fraction,
            "resampling": "Pillow LANCZOS, one floating source box for center_crop",
            "orientation": "EXIF transpose before geometry",
            "color": "RGBA uint8 alpha-composited on white, then RGB uint8 resize",
            "alpha_background": [255, 255, 255],
            "padding_background": [255, 255, 255],
            "output": "detached RGB BCHW float32, uint8 / 255, no VAE encode/decode",
            "cache_device": "cpu",
            "cache_size": self.cache_size,
            "require_sha256": self.require_sha256,
        }

    @property
    def last_metadata(self):
        return deepcopy(self._last_metadata)

    @property
    def cache_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for tensor, _ in self._cache.values())

    def clear_cache(self):
        self._cache.clear()

    def _entry(self, row):
        if not isinstance(row, Mapping):
            raise ValueError("reference metadata entries must be mappings with explicit id and image")
        identity, image = row.get("id"), row.get("image")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("reference metadata requires a nonempty string id")
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"reference metadata id={identity!r} requires an explicit image path")
        if not Path(image).is_absolute() and self.image_root is None:
            raise ValueError("relative reference image requires image_root; implicit cwd is unsupported")
        path = self._resolver(image)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = row.get("reference_image_sha256")
        if "reference_image_sha256" in row:
            if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
                raise ValueError("reference_image_sha256 must be exactly 64 hexadecimal characters")
            expected = expected.lower()
        elif self.require_sha256:
            raise ValueError(f"reference metadata id={identity!r} requires reference_image_sha256")
        return identity, path, expected

    def _load_one(self, row, *, height, width):
        identity, path, expected = self._entry(row)
        before = _file_identity(path)
        key = (str(path), before, height, width, self.resize_mode, self.max_crop_fraction)
        cached = self._cache.get(key)
        hit = cached is not None
        if cached is None:
            raw = path.read_bytes()
            if _file_identity(path) != before:
                raise RuntimeError(f"reference image changed while reading: {path}")
            digest = hashlib.sha256(raw).hexdigest()
            if expected is not None and digest != expected:
                raise ValueError(f"reference_image_sha256 mismatch for id={identity!r}: {path}")
            with Image.open(BytesIO(raw)) as source:
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError(f"reference images must have exactly one frame: {path}")
                raw_size, source_mode = list(source.size), source.mode
                orientation = source.getexif().get(274)
                rgba = ImageOps.exif_transpose(source).convert("RGBA")
                geometry = preprocessing_geometry(
                    rgba.size, (width, height), resize_mode=self.resize_mode,
                    max_crop_fraction=self.max_crop_fraction)
                # Composite first so invisible RGB values cannot leak through
                # resampling at alpha edges. This also handles palette tRNS.
                image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
                if self.resize_mode == "letterbox":
                    image = ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS,
                                         color=(255, 255, 255))
                elif self.resize_mode == "stretch":
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                else:
                    image = image.resize((width, height), Image.Resampling.LANCZOS,
                                         box=geometry["source_crop_box"])
                array = np.array(image, dtype=np.uint8, copy=True)
            # Evaluation can run under inference_mode before reward-gradient
            # construction. Such a warmup must not cache inference tensors.
            with torch.inference_mode(False), torch.no_grad():
                tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255).contiguous()
            details = {
                "image": str(path), "reference_image_sha256": digest,
                "raw_size": raw_size, "source_mode": source_mode,
                "exif_orientation": orientation, "geometry": geometry,
                "color": self.metadata["color"], "alpha_background": [255, 255, 255],
                "padding_background": [255, 255, 255],
            }
            cached = (tensor, details)
            if self.cache_size:
                # Remove obsolete versions of this file rather than retaining
                # stale targets alongside its updated bytes.
                for old_key in list(self._cache):
                    if old_key[0] == str(path) and old_key[1] != before:
                        del self._cache[old_key]
                self._cache[key] = cached
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        tensor, details = cached
        if expected is not None and details["reference_image_sha256"] != expected:
            raise ValueError(f"reference_image_sha256 mismatch for id={identity!r}: {path}")
        if _file_identity(path) != before:
            raise RuntimeError(f"reference image changed while preparing target: {path}")
        return tensor, {**deepcopy(details), "id": identity, "cache_hit": hit}

    def load_batch(self, metadata, *, height, width, device=None):
        """Return detached FP32 targets in exactly metadata order, including repeats."""
        self._last_metadata = []
        if (not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes))
                or len(metadata) == 0):
            raise ValueError("reference metadata must be a nonempty sequence of id/image mappings")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (height, width)):
            raise ValueError("target height and width must be positive integers")
        tensors, details = [], []
        for row in metadata:
            tensor, detail = self._load_one(row, height=height, width=width)
            tensors.append(tensor)
            details.append(detail)
        with torch.inference_mode(False), torch.no_grad():
            # stack allocates fresh storage even for one item and repeated IDs.
            batch = torch.stack(tensors).to(device=device, dtype=torch.float32).detach()
        self._last_metadata = details
        return batch

    def valid_mask(self, *, device=None):
        """Detached [B,1,H,W] content mask for the most recent loaded batch.

        Call under the same scorer lock as load_batch. This excludes letterbox
        padding without changing legacy rewards that intentionally score it.
        """
        if not self._last_metadata:
            raise ValueError("load_batch must run before requesting its valid mask")
        with torch.inference_mode(False), torch.no_grad():
            masks = []
            for row in self._last_metadata:
                geometry = row["geometry"]
                width, height = geometry["target_size"]
                mask = torch.zeros((1, height, width), dtype=torch.float32, device=device)
                left, top, right, bottom = geometry["output_content_box"]
                mask[:, top:bottom, left:right] = 1
                masks.append(mask)
            return torch.stack(masks)

    __call__ = load_batch
