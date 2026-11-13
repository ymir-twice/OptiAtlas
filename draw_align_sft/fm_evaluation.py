"""Fixed held-out flow loss, isolated from the training random streams.

This is a diagnostic on a declared sigma grid, not an estimate of the
model-specific training-timestep distribution. Every image gets equal weight.
Frozen text embeddings, VAE samples and Gaussian noise are cached once per
rank; all ranks execute the same number of forwards, including under FSDP.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from .algorithms.flow_matching import flow_matching_loss, flow_matching_sample
from .data import image_dimensions, load_pairs
from .image_preprocessing import preprocessing_geometry
from .runtime import checkpoint


def _options(config):
    options = config.get("evaluation", {})
    if not options.get("enabled", False):
        return None
    if config["algorithm"]["name"] != "flow_matching":
        raise ValueError("evaluation.enabled currently requires native flow_matching training")
    if not options.get("path"):
        raise ValueError("evaluation.path must name a separate held-out JSONL")
    count = options.get("samples_per_rank", 2)
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError("evaluation.samples_per_rank must be a positive integer")
    sigmas = [float(value) for value in options.get("sigmas", [0.1, 0.3, 0.5, 0.7, 0.9])]
    if not sigmas or any(not math.isfinite(value) or not 0 < value < 1 for value in sigmas):
        raise ValueError("evaluation.sigmas must be nonempty and strictly inside (0, 1)")
    if len(set(sigmas)) != len(sigmas):
        raise ValueError("evaluation.sigmas must not repeat values")
    return {"path": str(Path(options["path"]).expanduser().resolve()),
            "split": str(options.get("split", "validation")),
            "samples_per_rank": int(count), "seed": int(options.get("seed", 1729)),
            "sigmas": sigmas, "max_pixels": int(options.get("max_pixels") or config["data"]["max_pixels"])}


def prepare_evaluation_manifest(config, training_rows, world_size):
    """Validate held-out membership before loading model weights, on rank zero."""
    options = _options(config)
    if options is None:
        return None, []
    if options["path"] == str(Path(config["data"]["path"]).expanduser().resolve()):
        # A single manifest can contain correctly disjoint train/validation splits.
        if options["split"] == config["data"].get("split", "train"):
            raise ValueError("Fixed FM evaluation must use held-out examples, not the training split")
    rows, fingerprint = load_pairs(options["path"], split=options["split"], require_images=True,
                                   hash_images=True)
    train_ids = {row["id"] for row in training_rows}
    train_images = {row["image"] for row in training_rows}
    if any(row["id"] in train_ids or row["image"] in train_images for row in rows):
        raise ValueError("Fixed FM evaluation overlaps the training data by ID or image path")
    # Renaming or copying an image must not turn it into held-out data. Hashes
    # come from load_pairs, never from a user-declared sha256 metadata field.
    from .data import file_digest
    train_hashes = {row.get("_image_sha256") or file_digest(row["image"]) for row in training_rows}
    if any(row["_image_sha256"] in train_hashes for row in rows):
        raise ValueError("Fixed FM evaluation overlaps training image contents (SHA256)")
    count = options["samples_per_rank"] * world_size
    if len(rows) < count:
        raise ValueError(f"Fixed FM evaluation needs {count} distinct held-out rows ({world_size} ranks), "
                         f"but evaluation.path contains {len(rows)}; reduce evaluation.samples_per_rank")
    order = list(range(len(rows)))
    random.Random(options["seed"]).shuffle(order)
    selected = [rows[index] for index in order[:count]]
    protocol = {"version": 1, **options, "data_sha256": fingerprint, "samples": count,
                "world_size": world_size, "selected_ids": [row["id"] for row in selected],
                "averaging": "equal_images_then_equal_sigmas", "vae_and_noise": "fixed_sample_seed_plus_index"}
    model = config.get("model", {})
    mode = model.get("resize_mode", "letterbox")
    # Validate framing before loading weights. Keep the historical letterbox
    # v1 hash intact; alternative framing is a distinct evaluation target.
    geometry = preprocessing_geometry((32, 32), (32, 32), resize_mode=mode,
                                      max_crop_fraction=model.get("max_crop_fraction", 0.1))
    if mode != "letterbox":
        name = model.get("name")
        qwen = name == "qwen_image21"
        preprocessing = {
            "resize_mode": mode, "model_name": name,
            "source_color_mode": "RGBA" if qwen else "RGB" if name == "sensenova_u15" else "adapter_defined",
            "exif_transpose": True, "resampling": "Pillow.LANCZOS",
            "pixel_normalization": "float32_pixels_div_127.5_minus_1",
            "crop_box": "centered_float_source_box" if mode == "center_crop" else None,
        }
        if mode == "center_crop":
            preprocessing["max_crop_fraction"] = geometry["max_crop_fraction"]
        if qwen:
            preprocessing["alpha"] = "preserve_source_alpha_or_opaque_for_RGB"
            preprocessing["vae_posterior"] = "sample" if model.get("sample_vae_posterior", True) else "mode"
        protocol.update(version=2, preprocessing=preprocessing)
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol, selected


@contextmanager
def evaluation_state(policy, device, seed):
    """Restore exact submodule modes and CPU/current-device RNG, even on errors.

    Only this rank's CUDA generator is touched. get_rng_state_all/manual_seed_all
    would create unnecessary contexts on the other seven training GPUs.
    """
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in policy.modules()]
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            policy.eval()
            with torch.no_grad():
                yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        # Calling .train(previous) recursively would destroy mixed child modes.
        for module, was_training in modes:
            module.training = was_training


class FixedFMEvaluator:
    def __init__(self, adapter, config, device, training_rows, manifest):
        self.adapter, self.config, self.device = adapter, config, torch.device(device)
        self.manifest = manifest
        fresh_manifest, rows = checkpoint.rank_zero_call(
            lambda: prepare_evaluation_manifest(config, training_rows, checkpoint.world_size()), heartbeat=True)
        if fresh_manifest != manifest:
            raise ValueError("Fixed FM validation data/config changed after preflight")
        self.rows = [(index, row) for index, row in enumerate(rows)
                     if index % checkpoint.world_size() == checkpoint.rank()]
        self.cache = None

    def _prepare_cache(self):
        cache = []
        for index, row in self.rows:
            # A row's VAE/noise draws stay the same if rank assignment changes.
            with evaluation_state(self.adapter.policy, self.device, self.manifest["seed"] + index):
                height, width = image_dimensions(row["image"], self.manifest["max_pixels"],
                                                 int(getattr(self.adapter, "image_multiple", 32)))
                condition = self.adapter.encode_condition([row["caption"]])
                clean = self.adapter.encode_images([row["image"]], height, width).detach()
                noise = self.adapter.sample_noise(1, height, width).detach()
                if clean.shape != noise.shape or not torch.isfinite(clean).all() or not torch.isfinite(noise).all():
                    raise ValueError(f"Invalid fixed FM clean/noise state for {row['id']}")
                cache.append((condition, clean, noise))
        return cache

    def evaluate(self, *, step, epoch, reason, output, logger=None):
        started = time.monotonic()
        if logger is not None:
            logger.event("fm_evaluation_started", step=step, reason=reason,
                         samples=self.manifest["samples"], sigmas=self.manifest["sigmas"])
        with evaluation_state(self.adapter.policy, self.device, self.manifest["seed"] + checkpoint.rank()):
            if self.cache is None:
                # Propagate decode/tokenization failures before any DDP/FSDP forward.
                self.cache = checkpoint.all_ranks_call(self._prepare_cache)
            totals = torch.zeros(len(self.manifest["sigmas"]), dtype=torch.float64, device=self.device)
            for condition, clean, noise in self.cache:
                for index, value in enumerate(self.manifest["sigmas"]):
                    sigma = torch.tensor([value], dtype=torch.float32, device=self.device)
                    noisy, target = flow_matching_sample(clean, noise, sigma)
                    target_fn = getattr(self.adapter, "fm_training_target", None)
                    if target_fn is not None:
                        target = target_fn(clean, noisy, sigma)
                    with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                        enabled=self.config["training"]["precision"] == "bf16"):
                        prediction = self.adapter.predict_velocity(noisy, sigma, condition)
                        loss = flow_matching_loss(prediction, target)
                    totals[index] += loss.detach().double()
            if dist.is_initialized():
                dist.all_reduce(totals)
            totals /= self.manifest["samples"]
            if not torch.isfinite(totals).all():
                raise FloatingPointError("Nonfinite fixed FM evaluation loss")
            values = totals.cpu().tolist()
        record = {"time": datetime.now(timezone.utc).isoformat(), "step": int(step), "epoch": float(epoch),
                  "reason": reason, "loss/flow_matching_fixed": sum(values) / len(values),
                  **{f"loss/flow_matching_fixed/sigma_{sigma:g}": loss
                     for sigma, loss in zip(self.manifest["sigmas"], values)},
                  **{key: self.manifest[key] for key in ("samples", "samples_per_rank", "sigmas", "seed",
                                                        "data_sha256", "protocol_sha256", "selected_ids")},
                  "timing/evaluation_seconds": time.monotonic() - started}
        def publish():
            with (Path(output) / "evaluation.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if logger is not None:
                logger.event("fm_evaluation_finished", step=step, reason=reason,
                             loss=record["loss/flow_matching_fixed"],
                             seconds=record["timing/evaluation_seconds"])
        checkpoint.rank_zero_call(publish)
        return record
