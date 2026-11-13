"""Differentiable, frozen HPS v2.0/v2.1 with version-pinned checkpoints.

The architecture/tokenizer and score follow tgxs002/HPSv2 ``img_score.py``:
OpenCLIP ViT-H-14, standard GELU, normalized image/text feature dot product.
No logit_scale or factor of 100 is applied. The official light_augmentation=True
evaluation path preserves the full frame with longest-side resize and black
letterboxing. Here that preprocessing stays in floating-point tensors, retaining
image gradients (there is no PIL/uint8 round trip).
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


HPSV2_MODEL_NAME = "xswu/HPSv2"
HPSV2_REVISION = "697403c78157020a1ae59d23f111aa58ced35b0a"
HPSV2_SOURCE_REVISION = "866735ecaae999fa714bd9edfa05aa2672669ee3"
HPSV2_FILENAME = "HPS_v2_compressed.pt"
HPSV2_SHA256 = "07cb0d20626232204ca3a6ec53ff5984e7e53212a8a7b0e0ec2bac1bdaf125a1"
HPSV2_CHECKPOINTS = {
    "v2.0": {"filename": HPSV2_FILENAME, "sha256": HPSV2_SHA256, "bytes": 1972484501},
    "v2.1": {"filename": "HPS_v2.1_compressed.pt",
             "sha256": "c57a38fb4a2f7e7c15bf00da2ea377cdf165448b4dd1052a484c215a998c9837",
             "bytes": 1972490005},
}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def hpsv2_checkpoint_spec(hps_version="v2.0"):
    """Official version-to-file map; v2.0 remains the historical default."""
    if not isinstance(hps_version, str) or hps_version not in HPSV2_CHECKPOINTS:
        raise ValueError("HPSv2 hps_version must be explicitly v2.0 or v2.1")
    return dict(HPSV2_CHECKPOINTS[hps_version])


def _verify_v21_checkpoint(path, spec):
    """A renamed v2.0 file must never be represented as v2.1.

    Download preparation checks both versions. Runtime verification is mandatory
    for the newly opt-in v2.1 path, including callers bypassing preparation.
    The old v2.0 strict-state-dict loader keeps its previous API behavior.
    """
    if path.stat().st_size != spec["bytes"]:
        raise ValueError("HPSv2 v2.1 checkpoint differs from the official size/SHA256")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError("HPSv2 v2.1 checkpoint differs from the official size/SHA256")


def differentiable_pixels(images, image_size=224):
    """HPSv2's deterministic full-frame resize/black-pad/CLIP-normalize path.

    Like official ResizeMaxSize, dimensions use Python round. Padding is also
    applied when the longest side already equals 224, fixing the upstream
    scale==1 non-square edge case. Bicubic interpolation is antialiased and not
    clamped: torchvision's floating-point tensor resize can overshoot [0,1].
    """
    if (not torch.is_tensor(images) or images.ndim != 4 or images.shape[1] != 3
            or not images.is_floating_point() or min(images.shape) < 1):
        raise ValueError("HPSv2 images must be a nonempty floating-point [B,3,H,W] RGB tensor")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    height, width = images.shape[-2:]
    scale = image_size / max(height, width)
    resized = (max(1, round(height * scale)), max(1, round(width * scale)))
    pixels = images.float()
    if resized != (height, width):
        pixels = F.interpolate(pixels, size=resized, mode="bicubic", align_corners=False, antialias=True)
    pad_h, pad_w = image_size - resized[0], image_size - resized[1]
    pixels = F.pad(pixels, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    mean = pixels.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


def load_hpsv2_model(checkpoint_path, *, device="cpu", dtype=torch.float32, hps_version="v2.0"):
    """Load every encoder weight strictly; never silently use random/base weights.

    Each version's compressed.pt is a complete fp16 state dictionary, not an adapter.
    Building parameters on meta avoids allocating/initializing a second huge
    encoder. Nonpersistent attention-mask buffers remain initialized on CPU.
    """
    from accelerate import init_empty_weights
    from open_clip import CLIP, get_model_config

    spec = hpsv2_checkpoint_spec(hps_version)
    path = Path(checkpoint_path).expanduser()
    if path.name != spec["filename"] or not path.is_file():
        raise ValueError(f"HPSv2 {hps_version} requires a local {spec['filename']} checkpoint")
    if hps_version == "v2.1":
        _verify_v21_checkpoint(path, spec)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("HPSv2 checkpoint must contain a complete nonempty state_dict")
    if any(name.startswith("module.") for name in state):
        if not all(name.startswith("module.") for name in state):
            raise ValueError("HPSv2 checkpoint has inconsistent module. prefixes")
        state = {name.removeprefix("module."): value for name, value in state.items()}
    with init_empty_weights(include_buffers=False):
        model = CLIP(**get_model_config("ViT-H-14"), quick_gelu=False)
    expected = model.state_dict()
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    if missing or unexpected:
        raise ValueError(f"HPSv2 checkpoint keys mismatch: missing={sorted(missing)}, extra={sorted(unexpected)}")
    for name, parameter in expected.items():
        if not torch.is_tensor(state[name]) or state[name].shape != parameter.shape:
            raise ValueError(f"HPSv2 checkpoint shape mismatch: {name}")
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).requires_grad_(False).eval()


class HPSv2Reward:
    """Raw per-image normalized cosine with the corresponding text prompt."""

    requires_reference_images = False

    def __init__(self, checkpoint_path=None, *, device="cpu", dtype="float32",
                 microbatch_size=1, gradient_checkpointing=True, cache_size=512,
                 model=None, tokenizer=None, hps_version="v2.0"):
        spec = hpsv2_checkpoint_spec(hps_version)
        for name, value in (("microbatch_size", microbatch_size), ("cache_size", cache_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        if dtype not in dtypes:
            raise ValueError("HPSv2 dtype must be float32, bfloat16 or float16")
        if not isinstance(gradient_checkpointing, bool):
            raise ValueError("HPSv2 gradient_checkpointing must be boolean")
        self.device = torch.device(device)
        self.dtype = dtypes[dtype]
        injected = model is not None
        if model is None:
            if checkpoint_path is None:
                raise ValueError("HPSv2 requires an explicit local checkpoint_path")
            model = load_hpsv2_model(checkpoint_path, device=self.device, dtype=self.dtype,
                                    hps_version=hps_version)
        self.model = model.to(device=self.device, dtype=self.dtype).requires_grad_(False).eval()
        if tokenizer is None:
            from open_clip import get_tokenizer
            tokenizer = get_tokenizer("ViT-H-14")
        self.tokenizer = tokenizer
        self.microbatch_size = microbatch_size
        self.gradient_checkpointing = gradient_checkpointing
        self.cache_size = cache_size
        self._text_cache = OrderedDict()
        visual = getattr(self.model, "visual", None)
        if gradient_checkpointing:
            setter = getattr(visual, "set_grad_checkpointing", None)
            if setter is None:
                raise ValueError("HPSv2 image encoder does not support gradient checkpointing")
            setter(True)
        self.metadata = {
            "kind": "hpsv2", "version": hps_version, "model_name": HPSV2_MODEL_NAME,
            "revision": HPSV2_REVISION, "checkpoint_filename": spec["filename"],
            "official_source_revision": HPSV2_SOURCE_REVISION,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "expected_checkpoint_sha256": spec["sha256"], "injected_model": injected,
            "runtime_checkpoint_identity_verified": not injected and hps_version == "v2.1",
            "architecture": "OpenCLIP ViT-H-14; GELU; image 224; text context 77",
            "score": "raw normalized image/text feature cosine; no logit_scale or factor of 100",
            "conditioning": "generated image and original prompt, not a GT image similarity",
            "preprocessing": "full-frame longest-side bicubic resize, antialias=True; "
                             "centered black letterbox to 224; CLIP normalization; no crop/augmentation",
            "preprocessing_edge_case": "also pad non-square inputs whose longest side already equals 224",
            "text": "official OpenCLIP ViT-H-14 tokenizer; truncate to 77 tokens including start/end tokens",
            "image_mean": list(CLIP_MEAN), "image_std": list(CLIP_STD),
            "model_frozen": True, "model_dtype": dtype,
            "gradient_checkpointing": gradient_checkpointing, "microbatch_size": microbatch_size,
            "minimum": -1.0, "maximum": 1.0,
        }

    def _text_features(self, prompts):
        missing = list(dict.fromkeys(prompt for prompt in prompts if prompt not in self._text_cache))
        if missing:
            with torch.no_grad():
                tokens = self.tokenizer(missing).to(self.device)
                features = F.normalize(self.model.encode_text(tokens).float(), dim=-1)
            # Retain this call's embeddings even if its unique prompt count exceeds
            # cache_size, then evict only after building the returned batch.
            local = dict(zip(missing, features.detach().cpu()))
        else:
            local = {}
        result = []
        for prompt in prompts:
            value = local[prompt] if prompt in local else self._text_cache[prompt]
            result.append(value)
            self._text_cache[prompt] = value
            self._text_cache.move_to_end(prompt)
        while len(self._text_cache) > self.cache_size:
            self._text_cache.popitem(last=False)
        return torch.stack(result).to(self.device)

    def __call__(self, images, prompts, reference_images=None):
        if (not torch.is_tensor(images) or images.ndim != 4 or images.shape[1] != 3
                or not images.is_floating_point() or min(images.shape) < 1):
            raise ValueError("HPSv2 images must be a nonempty floating-point [B,3,H,W] RGB tensor")
        if (isinstance(prompts, str) or len(prompts) != images.shape[0]
                or not all(isinstance(prompt, str) for prompt in prompts)):
            raise ValueError("HPSv2 requires one string prompt per image")
        self.model.eval()
        scores = []
        with torch.autocast(device_type=self.device.type, enabled=False):
            text_features = self._text_features(prompts)
            for start in range(0, images.shape[0], self.microbatch_size):
                pixels = differentiable_pixels(images[start:start + self.microbatch_size].to(self.device))
                image_features = F.normalize(self.model.encode_image(pixels.to(self.dtype)).float(), dim=-1)
                scores.append((image_features * text_features[start:start + self.microbatch_size]).sum(-1))
        return torch.cat(scores).to(images.device)
