"""Reward extensions for the full DiffusionOPSD trainer.

The default is the vendored registry, including its seven public rewards,
weighted combinations, original CLIP token truncation, image preprocessing,
and learned-logit score scaling. The trainer retains the upstream independent
differentiable forwards for target construction. Nothing here replaces those
forwards with the earlier windowed CLIP cosine objective.

PickScore and Aesthetic use instance-local feature-output adapters for the
Transformers 4/5 API difference; their original scorer forwards are retained.
Explicitly selected ``custom`` or ``clip_similarity`` rewards share one frozen
scorer for endpoint, evaluation, and target calls.
The simple custom signature remains ``score(images, prompts,
reference_images=None) -> Tensor[B]``. Images retain their autograd graph;
the endpoint/evaluation wrapper alone disables gradient recording.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import threading
import weakref

import torch

from draw_align_sft.rewards.features import projected_features


_SCORERS = weakref.WeakValueDictionary()
_SCORERS_LOCK = threading.Lock()
_REFERENCE_NAMES = {"psnr", "mse", "dense_illum", "dino_dense", "dinov3_dense", "regional_luminance", "vjepa_dense", "ijepa_dense",
                    "dino_mse_luminance", "dinov3_mse_luminance", "dinov3_hpsv2_mse", "dinov2_hpsv2_mse", "ijepa_mse_luminance", "ijepa_mse_luminance_hpsv3"}
_EXTENSION_NAMES = {"custom", "clip_similarity"} | _REFERENCE_NAMES
_ORIGINAL_NAMES = {"original", "upstream", "mixed"}
_FEATURE_ADAPTER_NAMES = {"pickscore", "aesthetic"}


class _ProjectedFeatureModel(torch.nn.Module):
    """Adapt only this evaluator instance; preserve the vendor's full forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        return projected_features(self.model.get_image_features(*args, **kwargs))

    def get_text_features(self, *args, **kwargs):
        return projected_features(self.model.get_text_features(*args, **kwargs))


def load_compatible_original_scorer(kind, device):
    """Load an original PickScore/Aesthetic evaluator with portable features.

    Its original endpoint processor, text truncation, normalization, and score
    scaling remain untouched. The same instance is also suitable for the
    trainer's differentiable target forward.
    """
    if kind == "pickscore":
        from diffusionopsd.pickscore_scorer import PickScoreScorer

        scorer = PickScoreScorer(dtype=torch.float32, device=device)
        scorer.model = _ProjectedFeatureModel(scorer.model)
    elif kind == "aesthetic":
        from diffusionopsd.aesthetic_scorer import AestheticScorer

        scorer = AestheticScorer(dtype=torch.float32, device=device)
        scorer.clip = _ProjectedFeatureModel(scorer.clip)
    else:
        raise ValueError(f"No feature-output adapter needed for reward {kind!r}")
    return scorer.eval().requires_grad_(False)


def _build_original_reward_function(device, weights):
    from diffusionopsd.rewards import multi_score

    compatible_names = _FEATURE_ADAPTER_NAMES.intersection(weights)
    if not compatible_names:
        return multi_score(device, weights)
    # The unaffected rewards still execute their actual registry wrappers.
    # Compose those with the adapted evaluator instances, without mutating the
    # vendored registry or any Transformers class/process-wide state.
    original_weights = {name: weight for name, weight in weights.items() if name not in compatible_names}
    original = multi_score(device, original_weights) if original_weights else None
    scorers = {name: load_compatible_original_scorer(name, device)
               for name in weights if name in compatible_names}

    @torch.no_grad()
    def endpoint(images, prompts, metadata, only_strict=True):
        from PIL import Image

        details = original(images, prompts, metadata, only_strict=only_strict)[0] if original else {}
        for name, scorer in scorers.items():
            # Preserve the vendor registry's endpoint-only uint8/PIL conversion;
            # target construction calls the differentiable trainer helper.
            if name == "pickscore":
                pixels = images
                if torch.is_tensor(pixels):
                    arrays = (pixels * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                    pixels = [Image.fromarray(image) for image in arrays.transpose(0, 2, 3, 1)]
                details[name] = scorer(prompts, pixels)
            else:
                pixels = ((images * 255).round().clamp(0, 255).to(torch.uint8)
                          if torch.is_tensor(images)
                          else torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.uint8))
                details[name] = scorer(pixels)
        details = {name: details[name] for name in weights}
        totals = None
        for name, weight in weights.items():
            weighted = [weight * score for score in details[name]]
            totals = weighted if totals is None else [current + update for current, update in zip(totals, weighted)]
        details["avg"] = totals
        return details, {}

    return endpoint


def _get(value, key, default=None):
    if isinstance(value, Mapping) or hasattr(value, "get"):
        return value.get(key, default)
    return getattr(value, key, default)


def _plain(value):
    # Resolve OmegaConf values before building the cache key without requiring
    # every caller (including the original ml_collections config) to use Hydra.
    from omegaconf import OmegaConf

    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _settings(config):
    settings = _get(_get(config, "draw_align", {}), "reward", {})
    settings = _plain(settings)
    if not isinstance(settings, dict):
        raise TypeError("draw_align.reward must be a mapping")
    return settings


def reward_name(config):
    """Select original registry versus an explicitly requested extension."""
    name = str(_settings(config).get("name", "original"))
    if name not in _ORIGINAL_NAMES | _EXTENSION_NAMES:
        raise ValueError(
            f"Unknown draw_align.reward.name={name!r}; select original, mixed, "
            f"one of {sorted(_EXTENSION_NAMES | _ORIGINAL_NAMES)}. Original weights belong in reward_fn."
        )
    return name


class DrawAlignScorer:
    """One callable shared by endpoint and target paths, with no image detach."""

    def __init__(self, function, name, device, reference_loader=None, require_reference_sha256=False,
                 reference_exclude_padding=False, component_normalization="none", component_normalization_eps=1e-4):
        self.function = function
        self.name = name
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.reference_loader = reference_loader
        self.require_reference_sha256 = require_reference_sha256
        self.reference_exclude_padding = reference_exclude_padding
        from .group_rewards import normalization_options
        self.component_normalization, self.component_normalization_eps = normalization_options({
            "name": name, "component_normalization": component_normalization,
            "component_normalization_eps": component_normalization_eps,
        })
        if reference_exclude_padding and (reference_loader is None or not getattr(function, "supports_valid_mask", False)):
            raise ValueError("Selected reward does not support reference padding masks")
        self._lock = threading.RLock()
        # A custom module must remain frozen just like the public evaluators.
        if isinstance(function, torch.nn.Module):
            function.to(self.device).eval().requires_grad_(False)

    def __call__(self, images, prompts, metadata=None):
        # Upstream dispatches endpoints to a thread pool while sampling. A
        # shared custom evaluator (or CLIP's mutable text cache) need not be
        # thread-safe; retain async overlap but serialize calls to this model.
        with self._lock:
            return self._score(images, prompts, metadata)

    def _score(self, images, prompts, metadata=None):
        if self.component_normalization == "prompt_group":
            return self.score_components(images, prompts, metadata)["score"]
        reference = self._reference(images, prompts, metadata)
        scores = self.function(images, list(prompts), reference_images=reference, **self._mask_arguments(images))
        return self._validate_scores(scores, images)

    def score_components(self, images, prompts, metadata=None):
        """Same GT and model as the combined score, retaining image gradients."""
        components = self.score_raw_components(images, prompts, metadata)
        if self.component_normalization == "prompt_group":
            from .group_rewards import normalize_components
            return normalize_components(components, prompts, metadata, self.function.component_weights)
        return components

    def score_raw_components(self, images, prompts, metadata=None):
        """Raw endpoint components; cross-rank group statistics are fitted by the trainer."""
        with self._lock:
            reference = self._reference(images, prompts, metadata)
            function = getattr(self.function, "reward_components", None)
            if not callable(function):
                raise ValueError("Selected reward does not expose component scores")
            scores = function(images, list(prompts), reference_images=reference, **self._mask_arguments(images))
            return {name: self._validate_scores(value, images) for name, value in scores.items()}

    def _mask_arguments(self, images):
        if self.reference_exclude_padding:
            return {"valid_mask": self.reference_loader.valid_mask(device=images.device)}
        return {}

    def _reference(self, images, prompts, metadata):
        if not torch.is_tensor(images) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("reward images must be a floating Tensor[B,3,H,W]")
        if not images.is_floating_point() or images.shape[0] != len(prompts) or not len(prompts):
            raise ValueError("reward images and prompts must have equal nonzero batch sizes")
        reference = None
        if self.reference_loader is not None:
            if metadata is None or len(metadata) != len(prompts):
                raise ValueError("Reference reward requires explicit metadata for every generated image")
            if self.require_reference_sha256 and any(not row.get("reference_image_sha256") for row in metadata):
                raise ValueError("Reference reward requires reference_image_sha256 in each metadata row")
            reference = self.reference_loader.load_batch(
                metadata, height=images.shape[-2], width=images.shape[-1], device=images.device)
        return reference

    @staticmethod
    def _validate_scores(scores, images):
        if not torch.is_tensor(scores):
            raise TypeError("custom reward must return Tensor[B], not Python/API scores")
        if scores.ndim != 1 or scores.shape[0] != images.shape[0]:
            raise ValueError(f"reward must return Tensor[{images.shape[0]}], got {tuple(scores.shape)}")
        if not scores.is_floating_point() or not torch.isfinite(scores).all():
            raise ValueError("reward scores must be finite floating point values")
        if torch.is_grad_enabled() and images.requires_grad and not scores.requires_grad:
            raise RuntimeError(
                "Reward detached the image gradient. Use differentiable Tensor operations; "
                "do not use no_grad(), detach(), PIL, NumPy, or a remote score API in the hook."
            )
        return scores.to(device=images.device, dtype=torch.float32)


def load_draw_align_scorer(config, device):
    """Load the optional hook once for both endpoint and gradient evaluation.

    The weak cache only shares evaluators that still have live callers; it does
    not keep GPU models alive after a run finishes. Original rewards are loaded
    by the trainer's original-reward loader instead.
    """
    name = reward_name(config)
    if name not in _EXTENSION_NAMES:
        raise ValueError("Original rewards must use the upstream scorer loader")
    settings = _settings(config)
    output = _get(_get(config, "draw_align", {}), "output_dir")
    key = (str(torch.device(device)), json.dumps(settings, sort_keys=True),
           str(output) if name in _REFERENCE_NAMES and output else None)
    with _SCORERS_LOCK:
        cached = _SCORERS.get(key)
        if cached is not None:
            return cached
        from draw_align_sft.rewards import build_reward

        reward_config = {key: value for key, value in settings.items() if key != "name"}
        reward_config["kind"] = {"clip_similarity": "clip_cosine", "mse": "negative_mse"}.get(name, name)
        function = build_reward(reward_config, device=device)
        reference_loader = None
        if name in _REFERENCE_NAMES:
            from draw_align_sft.rewards.reference_images import ReferenceImageLoader
            reference_loader = ReferenceImageLoader(
                image_root=settings.get("image_root"),
                resize_mode=settings.get("reference_resize_mode", "stretch"),
                max_crop_fraction=settings.get("reference_max_crop_fraction", 0.1),
                cache_size=settings.get("reference_cache_size", 32),
                require_sha256=bool(settings.get("require_reference_sha256", False)),
            )
        scorer = DrawAlignScorer(function, name, device, reference_loader=reference_loader,
                                require_reference_sha256=bool(settings.get("require_reference_sha256", False)),
                                reference_exclude_padding=bool(settings.get("reference_exclude_padding", False)),
                                component_normalization=settings.get("component_normalization", "none"),
                                component_normalization_eps=settings.get("component_normalization_eps", 1e-4))
        if name in _REFERENCE_NAMES and int(os.environ.get("RANK", "0")) == 0:
            if output:
                from draw_align_sft.data import file_digest
                model_path = Path(settings.get("model_name") or settings.get("checkpoint_path") or "/nonexistent-reward-model").expanduser()
                files = (sorted(model_path.glob("*.safetensors")) + sorted(model_path.glob("*.json"))
                         if model_path.is_dir() else [])
                # JEPA metadata already binds the large checkpoint digest; do
                # not read the full 5GB file again for this receipt.
                receipt = {"reward": function.metadata, "reference": reference_loader.metadata,
                           "model_files_sha256": {path.name: file_digest(path) for path in files}}
                if scorer.component_normalization == "prompt_group":
                    from .group_rewards import normalization_description
                    receipt["component_normalization"] = normalization_description(scorer.component_normalization_eps)
                    # Explicitly distinguish new direct-sum experiments from
                    # historical component-normalized runs using the tracker.
                    if _get(_get(config, "train", {}), "advantage_normalization", "upstream") == "none":
                        receipt["summed_reward_normalization"] = "none"
                        receipt["advantage_normalization"] = "none; weighted sum passed through without centering or scaling"
                if name == "ijepa_mse_luminance_hpsv3":
                    hps_directory = Path(settings["hpsv3_model_path"]).expanduser()
                    receipt["hpsv3_model_files_sha256"] = {
                        path.name: file_digest(path) for path in sorted(hps_directory.iterdir())
                        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".jinja"}
                    }
                if name in {"dinov3_hpsv2_mse", "dinov2_hpsv2_mse"}:
                    from draw_align_sft.rewards.hpsv2 import hpsv2_checkpoint_spec
                    hps_directory = Path(settings["hpsv2_model_path"]).expanduser()
                    hps_version = settings.get("hpsv2_version", "v2.1" if name == "dinov2_hpsv2_mse" else "v2.0")
                    hps_filename = hpsv2_checkpoint_spec(hps_version)["filename"]
                    receipt["hpsv2_model_files_sha256"] = {
                        filename: file_digest(hps_directory / filename)
                        for filename in (hps_filename, "provenance.json")
                    }
                if settings.get("reference_exclude_padding", False):
                    receipt["reference"]["padding_reduction"] = "exclude padding; valid-pixel/patch-coverage weighted"
                destination = Path(output) / "reward_manifest.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and json.loads(destination.read_text()) != receipt:
                    raise ValueError("Reference reward model bytes or preprocessing changed in existing run")
                temporary = destination.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(receipt, indent=2) + "\n")
                temporary.replace(destination)
        _SCORERS[key] = scorer
        return scorer


def score_draw_align_grad(scorer, images01, prompts, metadata=None):
    """Target reward path. Autograd is deliberately left enabled by caller."""
    return scorer(images01, prompts, metadata=metadata)


def build_reward_functions(config, device):
    """Return upstream-compatible endpoint/evaluation functions.

    Both accept ``(images, prompts, metadata, only_strict=True)`` and return
    ``({reward_name: scores, 'avg': weighted_scores}, metadata)``. For originals
    unchanged scorers use the vendored registry; PickScore/Aesthetic additionally
    adapt the Transformers feature-output API while retaining their forwards.
    For extensions both functions close over the same evaluator also returned
    by ``load_draw_align_scorer`` for differentiable target construction.
    """
    name = reward_name(config)
    if name in _ORIGINAL_NAMES:
        weights = _get(config, "reward_fn")
        if weights is None:
            raise ValueError("Original rewards require a nonempty reward_fn mapping")
        return _build_original_reward_function(device, weights), _build_original_reward_function(device, weights)

    scorer = load_draw_align_scorer(config, device)

    @torch.no_grad()
    def endpoint(images, prompts, metadata, only_strict=True):
        del only_strict
        if name == "dense_illum":
            components = scorer.score_components(images, prompts, metadata=metadata)
            return {name: components["score"], "avg": components["score"],
                    "dino_dense": components["dino_dense"], "illumination": components["illumination"]}, {}
        if name in {"dino_dense", "dinov3_dense", "regional_luminance", "vjepa_dense", "ijepa_dense",
                    "dino_mse_luminance", "dinov3_mse_luminance", "dinov3_hpsv2_mse", "dinov2_hpsv2_mse", "ijepa_mse_luminance", "ijepa_mse_luminance_hpsv3"}:
            # Each thread sees only its rank's endpoint batch, often an incomplete
            # prompt group. Fit/normalize only after all ranks finish sampling.
            component_scorer = (scorer.score_raw_components if scorer.component_normalization == "prompt_group"
                                else scorer.score_components)
            components = component_scorer(images, prompts, metadata=metadata)
            return {**components, name: components["score"], "avg": components["score"]}, {}
        scores = scorer(images, prompts, metadata=metadata)
        return {name: scores, "avg": scores}, {}

    return endpoint, endpoint


__all__ = [
    "build_reward_functions", "load_draw_align_scorer", "score_draw_align_grad",
    "reward_name", "DrawAlignScorer", "load_compatible_original_scorer",
]
