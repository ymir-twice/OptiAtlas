"""Differentiable image rewards, constructed lazily from Hydra configuration."""

from importlib import import_module
from functools import partial


def build_reward(config, device=None):
    """Build a differentiable reward with the public reward signature.

Custom example: {kind: custom, target: my_module:score_images, kwargs: {...}}.
The function receives images, prompts, reference_images=None. It must preserve
image gradients during target construction; OPSD validates that path at runtime.
"""
    settings = dict(config)
    kind = settings.get("kind", settings.get("name", "clip"))
    if kind == "dinov2_hpsv2_mse":
        from .dinov2_hpsv2_mse import DINOV2_MODEL_NAME, DINOV2_REVISION, Dinov2HPSv2MSEReward
        return Dinov2HPSv2MSEReward(
            model_name=settings.get("model_name", DINOV2_MODEL_NAME),
            revision=settings.get("revision", DINOV2_REVISION),
            device=device or settings.get("device", "cpu"),
            image_size=settings.get("image_size", 336), microbatch_size=settings.get("microbatch_size", 1),
            local_files_only=bool(settings.get("local_files_only", True)),
            dino_weight=settings.get("dino_weight", 1 / 3),
            mse_weight=settings.get("mse_weight", 1 / 3), hpsv2_weight=settings.get("hpsv2_weight", 1 / 3),
            hpsv2_model_path=settings.get("hpsv2_model_path"),
            hpsv2_microbatch_size=settings.get("hpsv2_microbatch_size", 1),
            hpsv2_dtype=settings.get("hpsv2_dtype", "float32"),
            hpsv2_gradient_checkpointing=settings.get("hpsv2_gradient_checkpointing", True),
            hpsv2_version=settings.get("hpsv2_version", "v2.1"),
        )
    if kind == "dinov3_hpsv2_mse":
        from .dinov3_hpsv2_mse import Dinov3HPSv2MSEReward
        return Dinov3HPSv2MSEReward(
            model_name=settings.get("model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
            revision=settings.get("revision"), device=device or settings.get("device", "cpu"),
            image_size=settings.get("image_size", 336), microbatch_size=settings.get("microbatch_size", 1),
            local_files_only=bool(settings.get("local_files_only", True)),
            dino_weight=settings.get("dino_weight", 1.0), hpsv2_weight=settings.get("hpsv2_weight", 0.3),
            mse_weight=settings.get("mse_weight", 0.3), hpsv2_model_path=settings.get("hpsv2_model_path"),
            hpsv2_microbatch_size=settings.get("hpsv2_microbatch_size", 1),
            hpsv2_dtype=settings.get("hpsv2_dtype", "float32"),
            hpsv2_gradient_checkpointing=settings.get("hpsv2_gradient_checkpointing", True),
            hpsv2_version=settings.get("hpsv2_version", "v2.0"),
        )
    if kind == "ijepa_mse_luminance_hpsv3":
        from .ijepa_mse_luminance_hpsv3 import IJEPAMSELuminanceHPSv3Reward
        return IJEPAMSELuminanceHPSv3Reward(
            model_name=settings.get("model_name", "facebook/ijepa_vith14_1k"),
            revision=settings.get("revision"), device=device or settings.get("device", "cpu"),
            image_size=settings.get("image_size", 224), microbatch_size=settings.get("microbatch_size", 1),
            local_files_only=bool(settings.get("local_files_only", True)),
            ijepa_weight=settings.get("ijepa_weight", 1.0), mse_weight=settings.get("mse_weight", 0.4),
            luminance_weight=settings.get("luminance_weight", 0.4), grids=tuple(settings.get("grids", [1, 2, 4, 8])),
            hpsv3_model_path=settings.get("hpsv3_model_path"), hpsv3_weight=settings.get("hpsv3_weight", 0.4),
            hpsv3_microbatch_size=settings.get("hpsv3_microbatch_size", 1),
            hpsv3_dtype=settings.get("hpsv3_dtype", "bfloat16"),
            hpsv3_gradient_checkpointing=settings.get("hpsv3_gradient_checkpointing", True),
        )
    if kind == "ijepa_mse_luminance":
        from .ijepa_mse_luminance import IJEPAMSELuminanceReward
        return IJEPAMSELuminanceReward(
            model_name=settings.get("model_name", "facebook/ijepa_vith14_1k"),
            revision=settings.get("revision"), device=device or settings.get("device", "cpu"),
            image_size=settings.get("image_size", 224),
            microbatch_size=settings.get("microbatch_size", 1),
            local_files_only=bool(settings.get("local_files_only", False)),
            ijepa_weight=settings.get("ijepa_weight", 1.0),
            mse_weight=settings.get("mse_weight", 0.4),
            luminance_weight=settings.get("luminance_weight", 0.4),
            grids=tuple(settings.get("grids", [1, 2, 4, 8])),
        )
    if kind in {"dino_mse_luminance", "dinov3_mse_luminance"}:
        from .dino_mse_luminance import DinoMSELuminanceReward
        return DinoMSELuminanceReward(
            model_name=settings.get("model_name"),
            revision=settings.get("revision"), device=device or settings.get("device", "cpu"),
            image_size=settings.get("image_size", 336),
            microbatch_size=settings.get("microbatch_size", 1),
            local_files_only=bool(settings.get("local_files_only", False)),
            dino_weight=settings.get("dino_weight", 1.0),
            mse_weight=settings.get("mse_weight", 0.4),
            luminance_weight=settings.get("luminance_weight", 0.4),
            grids=tuple(settings.get("grids", [1, 2, 4, 8])),
            backbone_family="dinov3" if kind == "dinov3_mse_luminance" else "dinov2",
        )
    if kind == "regional_luminance":
        from .luminance import RegionalLuminanceReward
        return RegionalLuminanceReward(grids=tuple(settings.get("grids", [1, 2, 4, 8])))
    if kind in {"dino_dense", "dinov3_dense"}:
        from .dino_dense import DinoDenseReward
        return DinoDenseReward(
            model_name=settings.get("model_name"),
            revision=settings.get("revision"), device=device or settings.get("device", "cpu"),
            image_size=int(settings.get("image_size", 336)),
            microbatch_size=int(settings.get("microbatch_size", 1)),
            local_files_only=bool(settings.get("local_files_only", False)),
            backbone_family="dinov3" if kind == "dinov3_dense" else "dinov2",
        )
    if kind == "vjepa_dense":
        from .vjepa import VJEPADenseReward
        return VJEPADenseReward(
            checkpoint_path=settings.get("checkpoint_path"),
            device=device or settings.get("device", "cpu"),
            image_size=int(settings.get("image_size", 384)),
            microbatch_size=int(settings.get("microbatch_size", 1)),
        )
    if kind == "ijepa_dense":
        from .ijepa import IJEPADenseReward
        return IJEPADenseReward(
            model_name=settings.get("model_name", "facebook/ijepa_vith14_1k"),
            revision=settings.get("revision"), device=device or settings.get("device", "cpu"),
            image_size=int(settings.get("image_size", 224)),
            microbatch_size=int(settings.get("microbatch_size", 1)),
            local_files_only=bool(settings.get("local_files_only", False)),
        )
    if kind == "dense_illum":
        from .dense_illum import DenseIllumReward
        return DenseIllumReward(
            model_name=settings.get("model_name", "facebook/dinov2-base"),
            revision=settings.get("revision"), device=device or settings.get("device", "cpu"),
            image_size=int(settings.get("image_size", 224)),
            illumination_weight=float(settings.get("illumination_weight", 0.2)),
            sigmas=tuple(settings.get("sigmas", [1, 2, 4])),
            microbatch_size=int(settings.get("microbatch_size", 2)),
            local_files_only=bool(settings.get("local_files_only", False)),
        )
    if kind in {"mse", "negative_mse"}:
        from .mse import MSEReward
        return MSEReward()
    if kind == "psnr":
        from .psnr import PSNRReward
        return PSNRReward(mse_floor=settings.get("mse_floor", 1e-8))
    if kind in {"clip", "clip_cosine"}:
        from .clip import CLIPScoreReward
        return CLIPScoreReward(
            model_name=settings.get("model_name", "openai/clip-vit-large-patch14"),
            revision=settings.get("revision"),
            device=device or settings.get("device", "cpu"),
            text_strategy=settings.get("text_strategy", "chunk_mean"),
            window_overlap=int(settings.get("window_overlap", 0)),
            image_strategy=settings.get("image_strategy", "full_resize"),
            text_batch_size=int(settings.get("text_batch_size", 32)),
            cache_size=int(settings.get("cache_size", 512)),
            local_files_only=bool(settings.get("local_files_only", False)),
        )
    if kind == "custom":
        target = settings.get("target", "")
        if ":" in target:
            module, attribute = target.split(":", 1)
        else:
            module, _, attribute = target.rpartition(".")
        if not module or not attribute:
            raise ValueError("custom reward target must be module:function")
        function = getattr(import_module(module), attribute)
        if not callable(function):
            raise TypeError("custom reward target must be callable")
        return partial(function, **dict(settings.get("kwargs", {})))
    raise ValueError(f"unknown reward kind: {kind}")


__all__ = ["build_reward"]
