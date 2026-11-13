"""Native, frozen T2I inference for base, FM and OPSD evaluation checkpoints.

OPSD exports contain the entire updated adapter, including its FM warm start.
They replace that adapter: never stack the initial FM LoRA a second time. The
selected export is checked against both its completion hashes and saved EMA.
No optimizer, training forward patch, reward or training sampler is loaded.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

from .fm_image_provider import LocalFMCheckpointProvider, _DRAW_ROOT, _digest, _inspect, _json


_FAMILIES = {"qwen_image21", "sensenova_u15"}
_NATIVE = _DRAW_ROOT / "vendor/sensenova_validated/sensenova_u1"


def _source_digest(root):
    root = Path(root).expanduser().resolve()
    if not (root / "__init__.py").is_file():
        raise ValueError(f"Missing native SenseNova source package: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(_digest(path).encode())
    return digest.hexdigest()


def _base_path(model_path, expected=None):
    if not model_path or not Path(model_path).expanduser().is_dir():
        raise ValueError(f"Missing original local generator base model: {model_path}")
    result = str(Path(model_path).expanduser().resolve())
    if expected and result != str(Path(expected).expanduser().resolve()):
        raise ValueError("Requested generator base differs from the adapter's original base")
    return result


def _sensenova_config_digest(configuration):
    """Compare architecture without changing inference's active KV cache.

    The saved training adapter identity contains top-level use_cache=False,
    explicitly assigned by SenseNovaTrainAdapter. Normalize only the copied
    identity dictionary to that value; do not disable native inference caching.
    """
    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()
                    if key not in {"_name_or_path", "transformers_version", "dtype", "torch_dtype"}}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value
    config = canonical(configuration)
    config["use_cache"] = False
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def _checked_file(checkpoint, complete, name, *, payload=False):
    path = checkpoint / name
    checksum = complete.get("files_sha256", {}).get(name)
    if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", str(checksum)):
        raise ValueError(f"Missing checkpoint file or integrity checksum: {name}")
    if not payload and _digest(path) != checksum:
        raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return str(checksum)


def _lora_settings(metadata, family):
    """Accept only the ordinary alpha/r adapter actually used by these runs."""
    if family == "qwen_image21":
        if metadata.get("peft_type") != "LORA" or metadata.get("init_lora_weights", True) not in (True, "gaussian"):
            raise ValueError("Expected standard Qwen PEFT LoRA, without a modified base")
    elif metadata.get("format") != "draw_align_sensenova_lora" or metadata.get("format_version") != 1:
        raise ValueError("Expected the native Draw-Align SenseNova LoRA format")
    if metadata.get("bias", "none") != "none":
        raise ValueError("Base-bias adapters are not supported")
    unsupported = ("use_dora", "use_rslora", "lora_bias", "fan_in_fan_out", "modules_to_save",
                   "rank_pattern", "alpha_pattern", "layer_replication", "target_parameters",
                   "trainable_token_indices", "use_qalora", "use_bdlora", "alora_invocation_tokens",
                   "arrow_config", "corda_config", "eva_config", "loftq_config", "kasa_config",
                   "lora_ga_config", "monteclora_config", "velora_config", "megatron_config",
                   "exclude_modules", "layers_pattern", "layers_to_transform")
    if any(metadata.get(key) for key in unsupported):
        raise ValueError("Unsupported nonstandard or partially targeted LoRA configuration")
    import math
    rank = metadata.get("r", metadata.get("rank"))
    alpha = metadata.get("lora_alpha", metadata.get("alpha"))
    dropout = metadata.get("lora_dropout", metadata.get("dropout", 0.0))
    if (type(rank) is not int or rank < 1 or type(alpha) not in (int, float)
            or not math.isfinite(alpha) or alpha <= 0 or type(dropout) not in (int, float)
            or not 0 <= dropout < 1):
        raise ValueError("Invalid LoRA rank/alpha/dropout")
    targets = metadata.get("target_modules")
    if not isinstance(targets, list) or not targets or not all(isinstance(x, str) and x for x in targets):
        raise ValueError("Missing exact LoRA target modules")
    return {"rank": rank, "alpha": alpha, "dropout": dropout, "target_modules": targets}


def _opsd_inspect(path, model_family, model_path=None, model_code_root=None, expected_data_epochs=None):
    export = Path(path).expanduser().resolve()
    checkpoint = export.parent
    if export.name != "lora" or not re.fullmatch(r"checkpoint-[0-9]+", checkpoint.name) or checkpoint.parent.name != "checkpoints":
        raise ValueError("OPSD checkpoint_path must select checkpoints/checkpoint-N/lora")
    run = checkpoint.parent.parent
    config, upstream, complete = _json(run / "resolved_config.json"), _json(run / "upstream_config.json"), _json(checkpoint / "complete.json")
    if (config.get("algorithm", {}).get("name") != "diffusion_opsd" or config.get("finetune", {}).get("mode") != "lora"
            or config.get("model", {}).get("name") != model_family):
        raise ValueError("Expected a DiffusionOPSD LoRA run for the specified generator family")
    step = int(checkpoint.name.split("-")[1])
    if (complete.get("format") != 1 or complete.get("native_policy") is not True
            or complete.get("require_ema") is not True or complete.get("global_step") != step):
        raise ValueError("Invalid OPSD completion marker or missing evaluation EMA")
    _checked_file(checkpoint, complete, "trainer_state.json")
    state = _json(checkpoint / "trainer_state.json")
    if any(state.get(key) != complete.get(key) for key in ("global_step", "epoch_completed")):
        raise ValueError("OPSD trainer position differs from completion marker")
    coverage = []
    with (run / "coverage.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line) if line.strip() else {}
            if row.get("global_step") == step:
                coverage.append(row)
    if (len(coverage) != 1 or coverage[0].get("status") != "optimizer_completed"
            or type(coverage[0].get("data_epochs_completed")) is not int):
        raise ValueError("Missing unambiguous data-epoch coverage for the selected OPSD step")
    coverage = coverage[0]
    if expected_data_epochs is not None and (coverage["data_epochs_completed"] != expected_data_epochs
            or coverage.get("records_in_current_data_epoch") != 0 or coverage.get("coverage_fraction") != 1.0):
        raise ValueError(f"Selected OPSD checkpoint is not exactly {expected_data_epochs} completed data epochs")
    if model_family == "qwen_image21":
        meta_file, adapter_file, tensor_file = "lora/draw_align_export.json", "lora/peft_config/adapter_config.json", "lora/pytorch_lora_weights.safetensors"
    else:
        meta_file = adapter_file = "lora/adapter_config.json"
        tensor_file = "lora/adapter_model.safetensors"
    for file in {meta_file, adapter_file}:
        _checked_file(checkpoint, complete, file)
    for file in (tensor_file, "policy.pt", "ema.pt"):
        _checked_file(checkpoint, complete, file, payload=True)
    meta, adapter = _json(checkpoint / meta_file), _json(checkpoint / adapter_file)
    effective = copy.deepcopy(upstream["draw_align"]["model"])
    if effective.get("name") != model_family or upstream.get("train", {}).get("ema") is not True:
        raise ValueError("Effective OPSD model or EMA mode conflicts with selected export")
    if meta.get("mode") != "lora":
        raise ValueError("Expected an OPSD LoRA export")
    base = _base_path(model_path or effective.get("model_path"), meta.get("base_model"))
    _base_path(effective.get("model_path"), base)
    if model_family == "qwen_image21" and (meta.get("initial_transformer_path") != effective.get("transformer_path")
                                           or meta.get("revision") != effective.get("revision")):
        raise ValueError("Qwen export differs from its effective base transformer or revision")
    source = str(Path(model_code_root or _NATIVE).expanduser().resolve())
    source_hash = _source_digest(source) if model_family == "sensenova_u15" else None
    if source_hash is not None and source_hash != meta.get("source_fingerprint"):
        raise ValueError("Native SenseNova source differs from the OPSD training source")
    effective.update(model_path=base, model_code_root=source, lora_path=None)
    config = {**config, "model": effective, "finetune": {"mode": "lora", "lora": _lora_settings(adapter, model_family)}}
    identity = {"provider": "draw_align_checkpoint", "checkpoint_kind": "opsd_ema", "model_family": model_family,
                "checkpoint_path": str(export), "step": step, "base_model": base,
                "data_epochs_completed": coverage["data_epochs_completed"],
                "records_in_current_data_epoch": coverage.get("records_in_current_data_epoch"),
                "training_records": coverage.get("training_records"),
                "initial_lora_path_provenance": meta.get("initial_lora_path"),
                "adapter_semantics": "complete_updated_adapter_not_additive_delta",
                "state_sha256": complete["files_sha256"][tensor_file],
                "complete_sha256": _digest(checkpoint / "complete.json"),
                "resolved_config_sha256": _digest(run / "resolved_config.json"),
                "upstream_config_sha256": _digest(run / "upstream_config.json"),
                "native_source_sha256": source_hash, "optimizer_loaded": False}
    return config, meta, complete, identity, tensor_file


def checkpoint_identity(image):
    """Preflight metadata and expose an immutable generation/resume identity."""
    family = image["model_family"]
    if family not in _FAMILIES:
        raise ValueError(f"Unsupported generator family: {family}")
    path = image.get("checkpoint_path")
    if path:
        from .full_fm_image_provider import inspect_full_fm_export, is_full_fm_export
        if is_full_fm_export(path):
            return inspect_full_fm_export(path, model_family=family, model_path=image.get("model_path"),
                                          expected_data_epochs=image.get("expected_data_epochs"))[-1]
    if path and Path(path).name == "lora":
        return _opsd_inspect(path, family, image.get("model_path"), image.get("model_code_root"), image.get("expected_data_epochs"))[3]
    if path:
        identity = _inspect(path)[-1]
        if identity["model_family"] != family:
            raise ValueError("FM checkpoint belongs to a different generator family")
        _base_path(image.get("model_path") or identity["base_model"], identity["base_model"])
        if image.get("expected_data_epochs") is not None:
            # Epoch assertions are opt-in: old final-checkpoint identities stay
            # unchanged, while new comparisons bind the exact training position.
            from .fm_image_provider import checkpoint_training_metadata
            identity.update(checkpoint_training_metadata(path, expected_data_epochs=image["expected_data_epochs"]))
        return {**identity, "provider": "draw_align_checkpoint", "checkpoint_kind": "flow_matching"}
    base = _base_path(image.get("model_path"))
    configuration = Path(base) / ("model_index.json" if family == "qwen_image21" else "config.json")
    source_hash = _source_digest(image.get("model_code_root") or _NATIVE) if family == "sensenova_u15" else None
    return {"provider": "draw_align_checkpoint", "checkpoint_kind": "base", "model_family": family,
            "base_model": base, "checkpoint_path": None, "base_config_sha256": _digest(configuration),
            "native_source_sha256": source_hash, "optimizer_loaded": False}


class LocalCheckpointProvider(LocalFMCheckpointProvider):
    """Official inference samplers for base, adapter and full-transformer weights."""
    def __init__(self, model_family, model_path=None, checkpoint_path=None, device="cuda:0", dtype="bf16",
                 size="1024x1024", seed=42, num_steps=None, cfg_scale=None, timestep_shift=3.0,
                 model_code_root=None, attn_backend="sdpa", expected_data_epochs=None):
        options = dict(model_family=model_family, model_path=model_path, checkpoint_path=checkpoint_path,
                       model_code_root=model_code_root, expected_data_epochs=expected_data_epochs)
        identity = checkpoint_identity(options)
        self.checkpoint_kind = identity["checkpoint_kind"]
        if self.checkpoint_kind == "flow_matching_full":
            from .full_fm_image_provider import LocalFullFMCheckpointProvider
            LocalFullFMCheckpointProvider.__init__(
                self, checkpoint_path, model_family=model_family, model_path=model_path, device=device, dtype=dtype,
                size=size, seed=seed, num_steps=num_steps, cfg_scale=cfg_scale, timestep_shift=timestep_shift,
                model_code_root=model_code_root, attn_backend=attn_backend, expected_data_epochs=expected_data_epochs)
            return
        if self.checkpoint_kind == "flow_matching":
            super().__init__(checkpoint_path, device, dtype, size, seed, num_steps, cfg_scale,
                             timestep_shift, model_code_root, attn_backend)
            self._checkpoint_identity = identity
            return
        self._checkpoint_identity, self.model_family = identity, model_family
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        if self.checkpoint_kind == "opsd_ema":
            self.config, self.manifest, self.complete, self._checkpoint_identity, self._tensor_file = _opsd_inspect(
                checkpoint_path, model_family, model_path, model_code_root, expected_data_epochs)
        else:
            self.config = {"model": {"model_path": identity["base_model"], "model_code_root": str(model_code_root or _NATIVE)}}
        self._configure_inference(device, dtype, size, seed, num_steps, cfg_scale,
                                  timestep_shift, model_code_root, attn_backend)

    @property
    def identity(self):
        value = super().identity
        if self.checkpoint_kind in {"base", "flow_matching_full"}:
            value["lora_precision"] = None
        return value

    def _load_weights(self):
        if self.checkpoint_kind == "flow_matching":
            return super()._load_weights()
        if self.checkpoint_kind != "opsd_ema":
            raise ValueError("Base generator has no adapter weights")
        import torch
        from safetensors.torch import load_file
        checkpoint = self.checkpoint_path.parent
        for name in (self._tensor_file, "policy.pt", "ema.pt"):
            _checked_file(checkpoint, self.complete, name)
        raw = load_file(str(checkpoint / self._tensor_file), device="cpu")
        weights = {}
        for name, value in raw.items():
            if self.model_family == "qwen_image21":
                if not name.startswith("transformer.") or not re.search(r"\.lora_[AB]\.weight$", name):
                    raise ValueError(f"Invalid Qwen export tensor key: {name}")
                name = re.sub(r"(\.lora_[AB])\.weight$", r"\1.default.weight", name.removeprefix("transformer."))
            elif not name.startswith("base.") or not name.endswith((".lora_A", ".lora_B")):
                raise ValueError(f"Invalid SenseNova export tensor key: {name}")
            if name in weights or value.dtype != torch.float32 or value.ndim != 2 or not torch.isfinite(value).all():
                raise ValueError(f"Invalid FP32 LoRA matrix: {name}")
            weights[name] = value
        policy = torch.load(checkpoint / "policy.pt", map_location="cpu", weights_only=True, mmap=True)
        ema = torch.load(checkpoint / "ema.pt", map_location="cpu", weights_only=True, mmap=True)
        names = policy.get("names", [])
        averages = ema.get("ema_parameters", [])
        if (not weights or policy.get("version") not in (1, 2) or policy.get("mode") != "lora"
                or policy.get("active_adapter") != "default" or len(set(names)) != len(names)
                or set(names) != set(weights) or len(averages) != len(names)):
            raise ValueError("OPSD export, default policy and saved EMA parameter banks disagree")
        for name, value in zip(names, averages):
            if value.dtype != torch.float32 or not torch.equal(weights[name], value):
                raise ValueError(f"OPSD export differs from saved evaluation EMA: {name}")
        if self.model_family == "sensenova_u15":
            modules = {name.rsplit(".lora_", 1)[0] for name in weights}
            if modules != set(self.manifest.get("modules", [])):
                raise ValueError("SenseNova exported module list differs from its tensor bank")
        return weights

    def _load(self):
        if self.checkpoint_kind == "flow_matching_full":
            from .full_fm_image_provider import LocalFullFMCheckpointProvider
            LocalFullFMCheckpointProvider._load(self)
            return
        if self.checkpoint_kind != "base":
            if self.client is not None:
                return
            super()._load()
            if self.checkpoint_kind == "opsd_ema" and self.model_family == "sensenova_u15":
                digest = _sensenova_config_digest(self.client.config.to_dict())
                if digest != self.manifest["base_config_sha256"]:
                    self.client, self.tokenizer = None, None
                    raise ValueError("Loaded SenseNova architecture differs from OPSD base metadata")
            return
        if self.client is not None:
            return
        import torch
        if str(_DRAW_ROOT) not in sys.path:
            sys.path.insert(0, str(_DRAW_ROOT))
        dtype = torch.bfloat16 if self.dtype in {"bf16", "bfloat16"} else torch.float32
        device = torch.device(self.device)
        if device.type == "cuda":
            device = torch.device("cuda:0") if device.index is None else device
            torch.cuda.set_device(device)
        base = self._checkpoint_identity["base_model"]
        if self.model_family == "qwen_image21":
            from diffusers import QwenImage21Pipeline
            self.client = QwenImage21Pipeline.from_pretrained(base, torch_dtype=dtype, local_files_only=True)
            self.client.to(device=device, dtype=dtype)
            for component in (self.client.transformer, self.client.text_encoder, self.client.vae):
                component.requires_grad_(False).eval()
        else:
            from draw_align_sft.legacy_sensenova.model import load_source
            package, fingerprint = load_source(self.model_code_root)
            if fingerprint != self._checkpoint_identity["native_source_sha256"]:
                raise ValueError("Native SenseNova source changed after preflight")
            package.set_attn_backend(self.attn_backend)
            loader = importlib.import_module("sensenova_u1.utils").load_model_and_tokenizer
            self.client, self.tokenizer = loader(base, dtype=dtype, device=device,
                                                 device_map={"": str(device)} if device.type == "cuda" else None)
            self.client.requires_grad_(False).eval()
