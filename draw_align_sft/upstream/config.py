"""Translate Hydra configuration into the original upstream ConfigDict."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .bootstrap import config_api, vendor_root
from .data import split_rows, materialize
from ..training_plan import resolve_opsd_training_plan


def world_size(config):
    launcher_world = (int(config.get("launcher", {}).get("nproc_per_node", 8))
                      * int(config.get("launcher", {}).get("nnodes", 1)))
    return int(os.environ.get("WORLD_SIZE") or config.get("distributed", {}).get("expected_world_size")
               or (1 if config.get("task") == "train" else launcher_world))


def _merge(target, values):
    for key, value in values.items():
        if isinstance(value, dict) and key in target and hasattr(target[key], "items"):
            _merge(target[key], value)
        else:
            target[key] = value


def reference_velocity_weight(opsd):
    """Validate the opt-in initial-policy MSE; this is not an exact KL."""
    value = opsd.get("reference_velocity_weight", 0.0)
    if isinstance(value, bool):
        raise ValueError("opsd.reference_velocity_weight must be finite and nonnegative")
    try:
        weight = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("opsd.reference_velocity_weight must be finite and nonnegative") from error
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("opsd.reference_velocity_weight must be finite and nonnegative")
    if opsd.get("reference_source", "initial") != "initial":
        raise ValueError("opsd.reference_source only supports the permanently frozen initial policy")
    if weight and not bool(int(opsd.get("opa", 1))):
        raise ValueError("reference_velocity_weight requires the OPA training branch")
    interval = opsd.get("reference_diagnostics_interval", 0)
    if type(interval) is not int or interval < 0:
        raise ValueError("opsd.reference_diagnostics_interval must be a nonnegative integer")
    if interval and not weight:
        raise ValueError("reference_diagnostics_interval requires positive reference_velocity_weight")
    return weight


def validate_component_normalization(reward, upstream):
    """Validate grouped OPSD before loading models or starting collectives."""
    from .group_rewards import normalization_description, normalization_options
    mode, epsilon = normalization_options(reward)
    advantage_mode = getattr(upstream, "train", {}).get("advantage_normalization", "upstream")
    if not isinstance(advantage_mode, str) or advantage_mode not in {"upstream", "none"}:
        raise ValueError("train.advantage_normalization must be upstream or none")
    if advantage_mode == "none" and mode != "prompt_group":
        raise ValueError("train.advantage_normalization=none requires prompt_group component normalization")
    if mode == "none":
        return {"mode": "none"}
    if not bool(int(upstream.opsd.get("opa", 0))):
        raise ValueError("prompt_group component normalization requires opsd.opa=1")
    if any(bool(int(upstream.opsd.get(key, 0))) for key in ("refine", "aux_mode", "atc_x0")):
        raise ValueError("prompt_group component normalization cannot be combined with legacy RI/ATC targets")
    if int(upstream.eval_freq) != 0:
        raise ValueError("prompt_group component normalization requires eval_freq=0; evaluation has no K-rollout groups")
    if not reward.get("require_reference_sha256", False):
        raise ValueError("prompt_group component normalization requires require_reference_sha256=true")
    if int(upstream.sample.num_image_per_prompt) < 2:
        raise ValueError("prompt_group component normalization requires at least two rollouts per prompt")
    if int(upstream.opsd.get("opa_mb", 6)) < 1:
        raise ValueError("opsd.opa_mb must be positive")
    return normalization_description(epsilon)


def _write_json(path, value):
    # Other ranks may still read legacy source provenance in this directory.
    # Publish a whole metadata document instead of truncating it in place.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def build_config(config, *, dataset_directory=None, training_records=None):
    from ml_collections import ConfigDict
    options = config["upstream"]
    api = config_api()
    # Keep the real library's recipe wherever its fixed K=24 grouping fits.
    # Small test topologies (notably two GPUs) cannot build that default at
    # all, even when callers provide a legal smaller K below. Seed those from
    # the canonical eight-GPU recipe, then validate the effective overrides
    # against the actual world size in preflight.
    try:
        upstream = api.sd3_reward_config("clipscore", n_gpus=world_size(config))
    except ValueError as error:
        if not str(error).startswith("Cannot satisfy 48x24 grouping with "):
            raise
        upstream = api.sd3_reward_config("clipscore", n_gpus=8)
    api.attach_opsd(upstream, api.sd35m_opsd_default_params())
    upstream.pretrained.model = config["model"]["model_path"]
    upstream.pretrained.revision = config["model"].get("revision") or ""
    upstream.use_lora = config["finetune"]["mode"] == "lora"
    upstream.mixed_precision = {"bf16": "bf16", "fp32": "no", "fp16": "fp16"}[config["training"]["precision"]]
    # Resolve the data-epoch budget only after all batch-size overrides below.
    upstream.num_epochs = int(options["num_epochs"]) if options.get("num_epochs") is not None else 1
    upstream.save_freq = int(options["save_freq"])
    upstream.eval_freq = int(options["eval_freq"])
    upstream.seed = options.get("seed")
    upstream.debug = bool(options.get("debug", False))
    upstream.reward_fn = ConfigDict(dict(options["reward_weights"]))
    upstream.save_dir = str(Path(config["output_dir"]).resolve())
    upstream.logdir = str(Path(upstream.save_dir) / "wandb")
    upstream.run_name = config.get("logging", {}).get("wandb", {}).get("name") or f"{config['model']['name']}_opsd"
    upstream.dataset = str(dataset_directory or Path(upstream.save_dir) / "upstream_data")
    upstream.resume_from = upstream.save_dir if config.get("resume") == "latest" else (config.get("resume") or "")
    _merge(upstream, options.get("overrides", {}))
    additional_steps = upstream.get("additional_checkpoint_steps", ())
    if (not isinstance(additional_steps, (list, tuple))
            or any(type(step) is not int or step < 1 for step in additional_steps)):
        raise ValueError("additional_checkpoint_steps must contain only positive integer optimizer steps")
    reference_velocity_weight(upstream.opsd)
    validation_enabled = options.get("validation", {}).get("enabled", True)
    if type(validation_enabled) is not bool:
        raise ValueError("upstream.validation.enabled must be a boolean")
    if not validation_enabled and int(upstream.eval_freq) != 0:
        raise ValueError("upstream.validation.enabled=false requires upstream.eval_freq=0")
    if type(upstream.sample.get("save_all_eval_images", False)) is not bool:
        raise ValueError("sample.save_all_eval_images must be a boolean")
    plan_options = dict(options)
    if "num_epochs" in options.get("overrides", {}):
        plan_options["num_epochs"] = options["overrides"]["num_epochs"]
    if training_records is None:
        training_records = len(split_rows(config)[0])
    plan = resolve_opsd_training_plan(plan_options, upstream.sample, training_records, world_size(config))
    if "prompt_sampling" not in upstream.sample and options.get("data_epochs") is not None:
        upstream.sample.prompt_sampling = "epoch"
    if plan["mode"] == "data_epochs":
        # Random replacement does not guarantee five complete dataset sweeps.
        sampling = upstream.sample.get("prompt_sampling", "epoch")
        if sampling not in {"random", "epoch"}:
            raise ValueError("sample.prompt_sampling must be random or epoch")
        if sampling != "epoch":
            raise ValueError("upstream.data_epochs requires sample.prompt_sampling=epoch; "
                             "use explicit upstream.num_epochs for random sampling")
        upstream.sample.prompt_sampling = "epoch"
    upstream.num_epochs = plan["outer_epochs"]
    upstream.draw_align_training_plan = ConfigDict(plan, type_safe=False)
    if dataset_directory is None and "dataset" not in options.get("overrides", {}):
        upstream.dataset = str(Path(upstream.save_dir) / "upstream_data")
    if config.get("resume") == "latest" and "resume_from" not in options.get("overrides", {}):
        upstream.resume_from = upstream.save_dir
    reward_name = config.get("reward", {}).get("name", "original")
    if (reward_name == "dinov2_hpsv2_mse"
            and config["reward"].get("component_normalization") == "prompt_group"
            and "advantage_normalization" not in upstream.train):
        # The new grouped mixture defaults to its literal weighted total even
        # outside the dedicated launcher. Historical reward defaults stay intact.
        upstream.train.advantage_normalization = "none"
    if reward_name in {"custom", "clip_similarity", "psnr", "mse", "dense_illum", "dino_dense", "dinov3_dense", "regional_luminance", "vjepa_dense", "ijepa_dense", "dino_mse_luminance", "dinov3_mse_luminance", "dinov3_hpsv2_mse", "dinov2_hpsv2_mse", "ijepa_mse_luminance", "ijepa_mse_luminance_hpsv3"}:
        upstream.opsd.opa_reward_kind = "draw_align"
    elif len(upstream.reward_fn) > 1:
        upstream.opsd.opa_reward_kind = "mixed"
    validate_component_normalization(config.get("reward", {}), upstream)
    effective = deepcopy(config)
    effective["model"]["model_path"] = upstream.pretrained.model
    if "revision" in effective["model"]:
        effective["model"]["revision"] = upstream.pretrained.revision or None
    if upstream.train.lora_path:
        # Both adapters now export LoRA directories. Legacy merged SenseNova
        # weights remain available through an explicit native-format opt-in.
        warm_format = upstream.train.get("warm_start_format", "lora")
        if warm_format not in {"lora", "native"}:
            raise ValueError("train.warm_start_format must be lora or native")
        if warm_format == "native":
            if config["model"]["name"] != "sensenova_u15":
                raise ValueError("train.warm_start_format=native is only supported for SenseNova")
            if effective["model"].get("lora_path"):
                raise ValueError("Native warm start conflicts with model.lora_path")
            effective["model"]["model_path"] = upstream.train.lora_path
        else:
            existing = effective["model"].get("lora_path")
            if existing and Path(existing).expanduser().resolve() != Path(upstream.train.lora_path).expanduser().resolve():
                raise ValueError("train.lora_path conflicts with model.lora_path")
            effective["model"]["lora_path"] = upstream.train.lora_path
    upstream.draw_align = ConfigDict(effective, type_safe=False)
    return upstream


def preflight(config):
    from .bootstrap import verify_vendor
    upstream_source = verify_vendor()
    if "dataset" in config["upstream"].get("overrides", {}):
        raise ValueError("upstream.overrides.dataset is an internal manifest directory; use data.path and upstream.validation.path")
    if config["training"]["precision"] not in {"bf16", "fp32"}:
        raise ValueError("Native Qwen/SenseNova adapters require training.precision=bf16 or fp32")
    if config["finetune"]["mode"] not in {"lora", "full"}:
        raise ValueError("finetune.mode must be lora or full")
    if config["distributed"]["mode"] not in {"auto", "ddp"}:
        raise ValueError("The upstream OPSD dual-policy trainer uses DDP; use distributed.mode=ddp. FM retains FSDP.")
    for key in ("batch_size", "num_epochs", "max_steps", "gradient_accumulation", "save_steps"):
        if config["training"].get(key) is not None:
            raise ValueError(f"Upstream OPSD does not use training.{key}; configure upstream.num_epochs / upstream.overrides.train instead")
    if config["training"].get("max_grad_norm", 1.0) != 1.0:
        raise ValueError("Upstream OPSD does not use training.max_grad_norm; configure upstream.overrides.train.max_grad_norm instead")
    if config["training"].get("export", True) is not True:
        raise ValueError("Upstream OPSD checkpoints always include the model export; training.export=false is only supported for FM")
    training, validation = split_rows(config)
    upstream = build_config(config, training_records=len(training))
    from ..optimizers import validate_optimizer_config
    from .optimizer import optimizer_options
    validate_optimizer_config(
        optimizer_options(upstream.train), finetune_mode=config["finetune"]["mode"],
        distributed_mode=config["distributed"]["mode"],
    )
    if config.get("reward", {}).get("name") in {"psnr", "mse", "dense_illum", "dino_dense", "dinov3_dense", "regional_luminance", "vjepa_dense", "ijepa_dense", "dino_mse_luminance", "dinov3_mse_luminance", "dinov3_hpsv2_mse", "dinov2_hpsv2_mse", "ijepa_mse_luminance", "ijepa_mse_luminance_hpsv3"}:
        from ..rewards import build_reward
        from ..rewards.reference_images import ReferenceImageLoader
        reward_options = config["reward"]
        if reward_options.get("name") in {"dense_illum", "dino_dense", "dinov3_dense", "vjepa_dense", "ijepa_dense", "dino_mse_luminance", "dinov3_mse_luminance", "dinov3_hpsv2_mse", "dinov2_hpsv2_mse", "ijepa_mse_luminance", "ijepa_mse_luminance_hpsv3"}:
            # Preflight must not download or replicate a vision model on CPU.
            weight = float(reward_options.get("illumination_weight", 0.2))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("illumination_weight must be finite and nonnegative")
            for key in ("image_size", "microbatch_size"):
                if int(reward_options.get(key, 224 if key == "image_size" else 2)) < 1:
                    raise ValueError(f"reward.{key} must be positive")
            name = reward_options["name"]
            if name in {"dino_dense", "dinov3_dense", "vjepa_dense", "dino_mse_luminance", "dinov3_mse_luminance", "dinov3_hpsv2_mse", "dinov2_hpsv2_mse"}:
                patch = 14 if name in {"dino_dense", "dino_mse_luminance", "dinov2_hpsv2_mse"} else 16
                default_size = 384 if name == "vjepa_dense" else 336
                if int(reward_options.get("image_size", default_size)) % patch:
                    raise ValueError(f"reward.image_size must be divisible by patch_size={patch}")
            if name in {"dino_mse_luminance", "dinov3_mse_luminance", "dinov3_hpsv2_mse", "dinov2_hpsv2_mse", "ijepa_mse_luminance", "ijepa_mse_luminance_hpsv3"}:
                feature_weight = "dino_weight" if name in {"dino_mse_luminance", "dinov3_mse_luminance", "dinov3_hpsv2_mse", "dinov2_hpsv2_mse"} else "ijepa_weight"
                weight_defaults = ((feature_weight, 1.0), ("mse_weight", 0.4), ("luminance_weight", 0.4))
                if name in {"dinov3_hpsv2_mse", "dinov2_hpsv2_mse"}:
                    weight_defaults = (("dino_weight", 1.0), ("mse_weight", 0.3), ("hpsv2_weight", 0.3))
                if name == "dinov2_hpsv2_mse":
                    weight_defaults = (("dino_weight", 1/3), ("mse_weight", 1/3), ("hpsv2_weight", 1/3))
                weights = [float(reward_options.get(key, default)) for key, default in weight_defaults]
                if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not any(weights):
                    raise ValueError("Mixed reward weights must be finite and nonnegative, with at least one positive")
                grids = reward_options.get("grids", [1, 2, 4, 8])
                if (not isinstance(grids, (list, tuple)) or not grids or
                        any(type(grid) is not int or grid < 1 for grid in grids) or len(set(grids)) != len(grids)):
                    raise ValueError("reward.grids must contain distinct positive integers")
            if name == "vjepa_dense" and reward_options.get("checkpoint_path"):
                if not Path(reward_options["checkpoint_path"]).is_file():
                    raise ValueError("V-JEPA reward checkpoint_path must be a readable local checkpoint")
            if name in {"dinov3_dense", "dinov3_mse_luminance", "dinov3_hpsv2_mse"} and reward_options.get("local_files_only", False):
                model_dir = Path(reward_options.get("model_name", "")).expanduser()
                if not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors").is_file():
                    raise ValueError("DINOv3 local reward requires readable config.json and model.safetensors")
                model_config = json.loads((model_dir / "config.json").read_text())
                if model_config.get("model_type") != "dinov3_vit" or model_config.get("patch_size") != 16:
                    raise ValueError("DINOv3 reward requires a dinov3_vit encoder with patch_size=16")
            if name == "dinov2_hpsv2_mse" and reward_options.get("local_files_only", False):
                model_dir = Path(reward_options.get("model_name", "")).expanduser()
                if not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors").is_file():
                    raise ValueError("DINOv2 local reward requires readable config.json and model.safetensors")
                model_config = json.loads((model_dir / "config.json").read_text())
                if model_config.get("model_type") != "dinov2" or model_config.get("patch_size") != 14:
                    raise ValueError("DINOv2 reward requires a dinov2 encoder with patch_size=14")
            if name in {"ijepa_dense", "ijepa_mse_luminance", "ijepa_mse_luminance_hpsv3"}:
                if int(reward_options.get("image_size", 224)) != 224:
                    raise ValueError("I-JEPA ViT-H/14 reward requires its fixed image_size=224")
                if reward_options.get("local_files_only", False):
                    model_dir = Path(reward_options.get("model_name", ""))
                    if not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors").is_file():
                        raise ValueError("I-JEPA local reward requires readable config.json and model.safetensors")
            if name in {"dinov3_hpsv2_mse", "dinov2_hpsv2_mse"}:
                from ..rewards.hpsv2 import hpsv2_checkpoint_spec
                hps_path = Path(reward_options.get("hpsv2_model_path") or "/nonexistent-hpsv2").expanduser()
                hps_version = reward_options.get("hpsv2_version", "v2.1" if name == "dinov2_hpsv2_mse" else "v2.0")
                hps_filename = hpsv2_checkpoint_spec(hps_version)["filename"]
                if not (hps_path / hps_filename).is_file() or not (hps_path / "provenance.json").is_file():
                    raise ValueError(f"HPSv2 requires its local {hps_version} checkpoint and provenance; "
                                     f"run scripts/download_hpsv2_reward.py --version {hps_version}")
                hps_provenance = json.loads((hps_path / "provenance.json").read_text())
                if (hps_provenance.get("model_id") != "xswu/HPSv2" or hps_provenance.get("hps_version") != hps_version
                        or hps_provenance.get("status") != "completed"):
                    raise ValueError(f"Expected verified official HPSv2 {hps_version} resources")
                for key in ("hpsv2_microbatch_size",):
                    value = reward_options.get(key, 1)
                    if type(value) is not int or value < 1:
                        raise ValueError(f"reward.{key} must be a positive integer")
                if type(reward_options.get("hpsv2_gradient_checkpointing", True)) is not bool:
                    raise ValueError("reward.hpsv2_gradient_checkpointing must be a boolean")
            if name == "ijepa_mse_luminance_hpsv3":
                weight = reward_options.get("hpsv3_weight", 0.4)
                if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                    raise ValueError("reward.hpsv3_weight must be finite and positive")
                microbatch = reward_options.get("hpsv3_microbatch_size", 1)
                if type(microbatch) is not int or microbatch != 1:
                    raise ValueError("reward.hpsv3_microbatch_size must be 1")
                if reward_options.get("hpsv3_dtype", "bfloat16") not in {"bfloat16", "float32"}:
                    raise ValueError("reward.hpsv3_dtype must be bfloat16 or float32")
                if type(reward_options.get("hpsv3_gradient_checkpointing", True)) is not bool:
                    raise ValueError("reward.hpsv3_gradient_checkpointing must be a boolean")
                hps_path = reward_options.get("hpsv3_model_path")
                required = ("HPSv3.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                            "preprocessor_config.json", "chat_template.json", "merges.txt", "vocab.json",
                            "provenance.json")
                if not hps_path or any(not (Path(hps_path).expanduser() / filename).is_file() for filename in required):
                    raise ValueError("HPSv3 requires its local checkpoint and Qwen2-VL tokenizer/config; "
                                     "run python scripts/download_hpsv3_reward.py before launching")
        else:
            build_reward(reward_options)
        ReferenceImageLoader(
            image_root=reward_options.get("image_root"),
            resize_mode=reward_options.get("reference_resize_mode", "stretch"),
            max_crop_fraction=reward_options.get("reference_max_crop_fraction", 0.1),
            cache_size=reward_options.get("reference_cache_size", 32),
        )
        if reward_options.get("reference_exclude_padding", False) and reward_options["name"] not in {
                "mse", "regional_luminance", "dino_dense", "dinov3_dense", "vjepa_dense", "ijepa_dense", "dino_mse_luminance", "dinov3_mse_luminance", "dinov3_hpsv2_mse", "dinov2_hpsv2_mse", "ijepa_mse_luminance", "ijepa_mse_luminance_hpsv3"}:
            raise ValueError("Selected reward does not support reference padding masks")
        if not bool(int(upstream.opsd.get("opa", 1))):
            raise ValueError("Reference rewards require opsd.opa=1 for explicit reference-image target gradients")
        for row in training + validation:
            if not row.get("image") or not Path(row["image"]).is_file():
                raise ValueError(f"Pixel reward requires a readable GT image for every train/eval row: {row['id']}")
            if reward_options.get("require_reference_sha256", False):
                digest = row.get("reference_image_sha256", "")
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError(f"Pixel reward requires a lowercase reference_image_sha256 for row {row['id']}")
    if upstream.mixed_precision not in {"bf16", "no"}:
        raise ValueError("Native model bridges support upstream mixed_precision bf16/no only")
    if bool(upstream.use_lora) != (config["finetune"]["mode"] == "lora"):
        raise ValueError("use_lora must agree with finetune.mode; set finetune=lora/full")
    if not upstream.per_prompt_stat_tracking:
        raise ValueError("The upstream OPSD trainer requires per_prompt_stat_tracking=True")
    sample, train = upstream.sample, upstream.train
    prompt_sampling = sample.get("prompt_sampling", "random")
    if prompt_sampling not in {"random", "epoch"}:
        raise ValueError("sample.prompt_sampling must be random or epoch")
    if prompt_sampling == "epoch" and upstream.seed is None:
        raise ValueError("sample.prompt_sampling=epoch requires a fixed upstream.seed for resumable data coverage")
    world = world_size(config)
    for name, value in {"num_epochs": upstream.num_epochs, "save_freq": upstream.save_freq,
        "sample.train_batch_size": sample.train_batch_size, "sample.num_batches_per_epoch": sample.num_batches_per_epoch,
        "sample.num_image_per_prompt": sample.num_image_per_prompt, "sample.test_batch_size": sample.test_batch_size,
        "train.batch_size": train.batch_size, "train.gradient_accumulation_steps": train.gradient_accumulation_steps,
        "train.num_inner_epochs": train.num_inner_epochs, "sample.num_steps": sample.num_steps}.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if int(upstream.eval_freq) < 0 or int(sample.num_steps) < 2:
        raise ValueError("eval_freq must be nonnegative and OPSD sampling needs >=2 steps")
    fraction = float(train.timestep_fraction)
    if not 0 < fraction <= 1 or int(int(sample.num_steps) * fraction) < 1:
        raise ValueError("train.timestep_fraction must be in (0,1] and retain at least one training timestep")
    k, global_batch = int(sample.num_image_per_prompt), int(sample.train_batch_size) * world
    if k < 2 or global_batch % k:
        raise ValueError("sample.train_batch_size * world_size must be divisible by K=num_image_per_prompt >=2")
    if len(training) < global_batch // k:
        raise ValueError(f"Need >= {global_batch // k} training rows for upstream K-repeat sampling; found {len(training)}")
    local_rollouts = int(sample.train_batch_size) * int(sample.num_batches_per_epoch)
    local_effective = int(train.batch_size) * int(train.gradient_accumulation_steps)
    if local_rollouts % local_effective:
        raise ValueError("Rollout samples must divide exactly into training minibatches and gradient accumulation")
    if int(upstream.opsd.get("opa_cert", 0)) and (sample.solver != "dpm2" or not sample.deterministic):
        raise ValueError("Upstream suffix certification requires deterministic DPM2 sampling")
    if config["model"]["name"] not in {"qwen_image21", "sensenova_u15"}:
        raise ValueError("The model bridge supports qwen_image21 and sensenova_u15; original SD3/ZImage launchers remain in vendor")
    fingerprint = hashlib.sha256(json.dumps([training, validation], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    report = {"backend": "upstream", "model": config["model"]["name"], "algorithm": "diffusion_opsd",
        **upstream_source,
        "records": len(training), "validation_records": len(validation), "data_sha256": fingerprint,
        "effective_model_config": upstream.draw_align.model.to_dict(),
        "world_size": world, "samples_per_outer_epoch": local_rollouts * world,
        "training_plan": upstream.draw_align_training_plan.to_dict(),
        "component_normalization": validate_component_normalization(config.get("reward", {}), upstream),
        "optimizer_steps_per_outer_epoch": local_rollouts // local_effective * int(train.num_inner_epochs),
        "upstream_config": {key: upstream[key].to_dict() if hasattr(upstream[key], "to_dict") else upstream[key]
                            for key in ("sample", "train", "opsd", "reward_fn", "num_epochs", "eval_freq", "save_freq", "resolution",
                                        "beta", "decay_type", "per_prompt_stat_tracking", "mixed_precision", "allow_tf32", "seed")},
        "gpu_validation": "not_performed_by_preflight"}
    return report, training


def run(config):
    from . import UPSTREAM_COMMIT
    from .bootstrap import install_vendor
    from .checkpoint import (
        load_checkpoint_provenance, load_run_provenance, resolve_resume_checkpoint, validate_resume_provenance,
    )
    info, _ = preflight(config)
    upstream = build_config(config, training_records=info["records"])
    rank = int(os.environ.get("RANK", "0"))
    root = Path(upstream.save_dir).resolve()
    provenance = {
        "upstream_commit": UPSTREAM_COMMIT, "vendor_root": str(vendor_root()),
        "model_config": config["model"], "finetune_config": config["finetune"],
        "reward_config": config["reward"], **info,
    }
    provenance_path = root / "upstream_provenance.json"
    existing_run = any((root / name).exists() for name in (
        "run_config.json", "resolved_config.json", "upstream_config.json", "upstream_provenance.json",
    )) or (root / "checkpoints").exists()
    # Only rank zero owns destination metadata. Other ranks can arrive after it
    # creates a new run and must not mistake that run for a pre-existing one.
    if rank == 0 and not upstream.resume_from and existing_run:
        raise FileExistsError("Existing OPSD run: use resume=latest or a new output_dir")
    if rank != 0 and not upstream.resume_from and existing_run:
        # A late rank may see metadata just published by its rank zero. It may
        # join that matching run, but must not materialize data into a different
        # run while rank zero is reporting a destination conflict.
        if not provenance_path.is_file():
            raise ValueError("Existing destination run has no upstream_provenance.json; use a new output_dir")
        validate_resume_provenance(load_run_provenance(provenance_path), provenance)
    if upstream.resume_from:
        # Resolve the effective override once, before writing anything to the
        # destination. An external checkpoint must be checked against its own
        # immutable snapshot (or, for legacy saves, its original run metadata).
        upstream.resume_from = resolve_resume_checkpoint(
            upstream.resume_from, require_ema=bool(upstream.train.ema),
            require_scaler=upstream.mixed_precision != "no", native_policy=True,
        )
        validate_resume_provenance(load_checkpoint_provenance(upstream.resume_from), provenance)
        if existing_run:
            if not provenance_path.is_file():
                raise ValueError("Existing destination run has no upstream_provenance.json; use a new output_dir")
            validate_resume_provenance(load_run_provenance(provenance_path), provenance)
    expected = config.get("distributed", {}).get("expected_world_size")
    actual = int(os.environ.get("WORLD_SIZE", "1"))
    if expected is not None and actual != int(expected):
        raise ValueError(f"Expected {expected} training workers, found {actual}")
    # Keep source provenance in every future checkpoint, including when the
    # checkpoint is copied away from the run directory.
    upstream.draw_align_provenance = provenance
    root.mkdir(parents=True, exist_ok=True)
    # Atomic deterministic manifests are safe if ranks enter simultaneously.
    materialize(config, root / "upstream_data")
    if rank == 0:
        # Publish provenance first so a late rank never sees a new run with
        # resolved metadata but no identity to validate.
        _write_json(provenance_path, provenance)
        _write_json(root / "resolved_config.json", config)
        _write_json(root / "upstream_config.json", upstream.to_dict())
    options = config.get("logging", {}).get("wandb", {})
    os.environ["WANDB_MODE"] = options.get("mode", "offline") if options.get("enabled") else "disabled"
    os.environ.setdefault("CODE_COMMIT", UPSTREAM_COMMIT)
    os.environ.setdefault("CODE_VARIANT", "draw_align_full_upstream_opsd")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    install_vendor()
    from . import trainer
    # The upstream profiler imports this sampler dynamically; route native
    # adapters through the same dispatcher used by train and evaluation.
    import diffusionopsd.diffusers_patch.pipeline_with_logprob as sampler_module
    sampler_module.pipeline_with_logprob = trainer.pipeline_with_logprob
    trainer.main(None, config_override=upstream)
    return {"status": "completed", "backend": "upstream", "output_dir": str(root)}
