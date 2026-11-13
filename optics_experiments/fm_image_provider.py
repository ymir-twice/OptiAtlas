"""Native T2I inference from a completed Draw-Align FM LoRA checkpoint.

Read the selected checkpoint itself, not the possibly later ``model/`` export.
Only adapter A/B tensors are installed. Optimizer/RNG/training forward methods
are never restored; both backends retain their official inference sampler.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys


_DRAW_ROOT = Path(__file__).resolve().parents[1] / "Draw-Align-SFT"


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _training_progress(checkpoint, config, manifest, complete, *, require_epoch=False):
    """Bind a completed step to its run events and, when needed, epoch plan."""
    step = complete.get("step")
    if type(step) is not int or step < 1:
        raise ValueError("Checkpoint step must be a positive integer")
    events = []
    with (checkpoint.parent.parent / "events.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("Invalid FM training event")
                events.append(event)
    finished = [(index, event) for index, event in enumerate(events) if event.get("event") == "training_finished"]
    if not finished:
        raise ValueError("FM checkpoint requires a successfully completed training run")
    final_index, final = finished[-1]
    final_step = final.get("step")
    if type(final_step) is not int or final_step < step:
        raise ValueError("Selected checkpoint is beyond the completed training step")
    # Preserve the established final-checkpoint path and identity for old runs
    # that predate epoch receipts. Intermediate selections need stronger proof.
    if step == final_step and not require_epoch:
        return None
    saved = [(index, event) for index, event in enumerate(events[:final_index])
             if event.get("event") == "checkpoint_saved" and event.get("step") == step]
    if not saved or any(type(event.get("step")) is not int
                        or not isinstance(event.get("path"), str)
                        or Path(event["path"]).expanduser().resolve() != checkpoint for _, event in saved):
        raise ValueError("Selected FM checkpoint has no matching completed checkpoint_saved receipt")
    plan = manifest.get("training_plan")
    fields = ("records", "world_size", "batch_size", "gradient_accumulation",
              "global_batch_size", "steps_per_epoch", "max_steps")
    if (not isinstance(plan, dict) or plan.get("mode") not in {"epochs", "steps"}
            or any(type(plan.get(key)) is not int or plan[key] < 1 for key in fields)):
        raise ValueError("FM checkpoint requires a valid original training epoch plan")
    batch = plan["world_size"] * plan["batch_size"] * plan["gradient_accumulation"]
    steps_per_epoch = (plan["records"] + batch - 1) // batch
    if (plan["global_batch_size"] != batch or plan["steps_per_epoch"] != steps_per_epoch
            or plan.get("padding_records_per_epoch") != steps_per_epoch * batch - plan["records"]
            or type(complete.get("world_size")) is not int or complete["world_size"] != plan["world_size"]
            or plan["max_steps"] != final_step):
        raise ValueError("FM checkpoint completion or batch/epoch plan is inconsistent")
    if plan["mode"] == "epochs" and (type(plan.get("num_epochs")) is not int or plan["num_epochs"] < 1
                                       or plan["max_steps"] != plan["num_epochs"] * steps_per_epoch):
        raise ValueError("FM training epoch budget does not match its optimizer steps")
    training = config.get("training")
    if (not isinstance(training, dict) or training != manifest.get("config", {}).get("training")
            or any(training.get(key) != plan[key] for key in ("batch_size", "gradient_accumulation"))
            or (training.get("max_steps") is not None and training["max_steps"] != plan["max_steps"])
            or (training.get("num_epochs") is not None and training["num_epochs"] != plan.get("num_epochs"))):
        raise ValueError("FM training configuration differs from its original epoch plan")
    starts = [event for event in events[:saved[-1][0]] if event.get("event") == "training_started"]
    if not any(event.get("training_plan") == plan and event.get("model") == manifest["model"]
               and event.get("algorithm") == "flow_matching" for event in starts):
        raise ValueError("FM training_started receipt does not confirm the selected checkpoint's epoch plan")
    return {"data_epochs_completed": step // steps_per_epoch,
            "steps_in_current_data_epoch": step % steps_per_epoch,
            "steps_per_epoch": steps_per_epoch, "training_final_step": final_step,
            "is_final_checkpoint": step == final_step}


def _inspect(checkpoint_path):
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if checkpoint.parent.name != "checkpoints" or not re.fullmatch(r"step_[0-9]{8}", checkpoint.name):
        raise ValueError("checkpoint_path must select a Draw-Align checkpoints/step_XXXXXXXX directory")
    run = checkpoint.parent.parent
    config, manifest, complete = (_json(run / "resolved_config.json"),
                                  _json(run / "training_manifest.json"),
                                  _json(checkpoint / "complete.json"))
    family = config.get("model", {}).get("name")
    if family not in {"qwen_image21", "sensenova_u15"}:
        raise ValueError("FM generator supports qwen_image21 and sensenova_u15")
    if config.get("algorithm", {}).get("name") != "flow_matching" or config.get("finetune", {}).get("mode") != "lora":
        raise ValueError("Expected a flow-matching LoRA run")
    if manifest.get("model") != family or manifest.get("algorithm") != "flow_matching":
        raise ValueError("Training manifest differs from the resolved model/algorithm")
    if any(config.get(key) != manifest.get("config", {}).get(key)
           for key in ("model", "finetune", "algorithm")):
        raise ValueError("Resolved model/LoRA configuration differs from the original training manifest")
    if (complete.get("sharded") is not False or complete.get("trainable_only") is not True
            or complete.get("identity") != manifest.get("identity")
            or complete.get("step") != int(checkpoint.name[5:])):
        raise ValueError("Checkpoint completion marker differs from the training identity/step or unsupported format")
    _training_progress(checkpoint, config, manifest, complete)
    checksum = complete.get("files_sha256", {}).get("state.pt", "")
    if not re.fullmatch(r"[0-9a-f]{64}", checksum) or not (checkpoint / "state.pt").is_file():
        raise ValueError("Checkpoint is missing its state.pt or valid integrity checksum")
    names = manifest.get("trainable_parameters", [])
    if not names or not all(isinstance(name, str) for name in names):
        raise ValueError("Training manifest is missing the exact trainable parameter bank")
    names = [name.removeprefix("module.") for name in names]
    if len(set(names)) != len(names):
        raise ValueError("Training manifest contains duplicate trainable parameter names")
    source = config["model"].get("pretrained_model_name_or_path", config["model"].get("model_path"))
    if not source or not Path(source).expanduser().is_dir():
        raise ValueError("The original local base model is missing")
    provenance = manifest.get("provenance", {}).get("model", {})
    if provenance.get("path") and Path(provenance["path"]).resolve() != Path(source).expanduser().resolve():
        raise ValueError("Base model path differs from the training provenance")
    identity = {"provider": "draw_align_fm", "model_family": family,
                "checkpoint_path": str(checkpoint), "step": complete["step"],
                "training_identity": manifest["identity"], "state_sha256": checksum,
                "resolved_config_sha256": _digest(run / "resolved_config.json"),
                "training_manifest_sha256": _digest(run / "training_manifest.json"),
                "base_model": str(Path(source).expanduser().resolve()),
                "base_model_sha256": provenance.get("sha256"),
                "native_source_sha256": manifest.get("provenance", {}).get("native_source", {}).get("loaded_sha256")
                    if manifest.get("provenance", {}).get("native_source") else None}
    return config, manifest, complete, names, identity


def checkpoint_identity(image):
    """Small-file preflight; tensor payload SHA256 is checked again at load."""
    path = image["checkpoint_path"] if isinstance(image, dict) else image
    return _inspect(path)[-1]


def checkpoint_training_metadata(image, *, expected_data_epochs=None):
    """Verify epoch evidence separately, leaving existing resume identities intact.

    ``expected_data_epochs`` requires an exact completed epoch boundary rather
    than accepting a checkpoint partway through the following data epoch.
    This reads JSON receipts only; the payload checksum remains a load check.
    """
    path = image["checkpoint_path"] if isinstance(image, dict) else image
    if expected_data_epochs is None and isinstance(image, dict):
        expected_data_epochs = image.get("expected_data_epochs")
    config, manifest, complete, _, identity = _inspect(path)
    progress = _training_progress(Path(identity["checkpoint_path"]), config, manifest, complete, require_epoch=True)
    if expected_data_epochs is not None:
        if type(expected_data_epochs) is not int or expected_data_epochs < 1:
            raise ValueError("expected_data_epochs must be a positive integer")
        if (progress["data_epochs_completed"] != expected_data_epochs
                or progress["steps_in_current_data_epoch"] != 0):
            raise ValueError(f"Selected FM checkpoint is not exactly {expected_data_epochs} completed data epochs")
    return progress


def _restore_parameters(policy, weights):
    """Restore exact FP32 masters after native loaders initialize LoRA modules."""
    import torch
    actual = {name: value for name, value in policy.named_parameters() if "lora_" in name}
    if set(actual) != set(weights):
        raise ValueError("Loaded model's complete LoRA key set differs from the selected checkpoint")
    for name, value in weights.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all() or actual[name].shape != value.shape:
            raise ValueError(f"Invalid FP32 LoRA tensor or shape: {name}")
    with torch.no_grad():
        for name, value in weights.items():
            actual[name].data = actual[name].data.float()
            actual[name].copy_(value.to(actual[name].device))
            if not torch.equal(actual[name].detach().cpu(), value):
                raise RuntimeError(f"Could not restore exact checkpoint tensor: {name}")
    policy.requires_grad_(False).eval()


def _install_sensenova_lora(model, weights, lora):
    """Attach trained generation-only LoRA without replacing native forwards."""
    import torch
    from draw_align_sft.models.sensenova import GenerationLoRALinear
    from draw_align_sft.legacy_sensenova.model import parameter_branch
    rank = int(lora.get("rank", lora.get("r", 16)))
    alpha = float(lora.get("alpha", lora.get("lora_alpha", rank)))
    dropout = float(lora.get("dropout", lora.get("lora_dropout", 0.0)))
    targets = lora.get("target_modules") or ["q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen", "gate_proj", "up_proj", "down_proj"]
    if isinstance(targets, str):
        targets = [targets]
    selected = [(name, module) for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and parameter_branch(name) == "generation"
                and ("all-linear" in targets or any(name == key or name.endswith("." + key) for key in targets))]
    expected = {"base." + name + suffix for name, _ in selected for suffix in (".lora_A", ".lora_B")}
    if not selected or expected != set(weights):
        raise ValueError("SenseNova checkpoint LoRA bank does not match all configured generation modules")
    for name, module in selected:
        for suffix, shape in ((".lora_A", (rank, module.in_features)), (".lora_B", (module.out_features, rank))):
            value = weights["base." + name + suffix]
            if tuple(value.shape) != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"Invalid SenseNova LoRA tensor: {name + suffix}")
    for name, module in selected:
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, GenerationLoRALinear(module, rank, alpha, dropout))
    _restore_parameters(model, {name.removeprefix("base."): value for name, value in weights.items()})


class LocalFMCheckpointProvider:
    def __init__(self, checkpoint_path, device="cuda:0", dtype="bf16", size="1024x1024",
                 seed=42, num_steps=None, cfg_scale=None, timestep_shift=3.0,
                 model_code_root=None, attn_backend="sdpa"):
        self.config, self.manifest, self.complete, self.parameter_names, self._checkpoint_identity = _inspect(checkpoint_path)
        self.checkpoint_path = Path(self._checkpoint_identity["checkpoint_path"])
        self.model_family = self._checkpoint_identity["model_family"]
        self._configure_inference(device, dtype, size, seed, num_steps, cfg_scale,
                                  timestep_shift, model_code_root, attn_backend)

    def _configure_inference(self, device, dtype, size, seed, num_steps, cfg_scale,
                             timestep_shift, model_code_root, attn_backend):
        """Shared native sampler settings for base, FM and OPSD checkpoints."""
        self.device, self.dtype, self.size = str(device), str(dtype), str(size)
        self.seed = int(seed)
        self.num_steps = int(num_steps if num_steps is not None else (40 if self.model_family == "qwen_image21" else 50))
        self.cfg_scale = float(cfg_scale if cfg_scale is not None else (1.0 if self.model_family == "qwen_image21" else 4.0))
        self.timestep_shift = float(timestep_shift)
        self.model_code_root = str(Path(model_code_root or self.config["model"].get("model_code_root", ".")).expanduser().resolve())
        self.attn_backend = str(attn_backend)
        self.client, self.tokenizer, self.last_metadata = None, None, None
        self.is_demo = False
        width, height = self._dimensions()
        if self.dtype not in {"bf16", "bfloat16", "fp32", "float32"}:
            raise ValueError("FM generation dtype must be bf16 or fp32")
        if not 1 <= self.num_steps <= 1000 or not math.isfinite(self.cfg_scale) or self.cfg_scale < 1:
            raise ValueError("Require positive sampling steps and finite cfg_scale >= 1")
        if not math.isfinite(self.timestep_shift) or self.timestep_shift <= 0:
            raise ValueError("timestep_shift must be positive and finite")
        self._apply_seed(self.seed)

    def _dimensions(self):
        width, height = map(int, self.size.lower().split("x"))
        if min(width, height) < 32 or width % 32 or height % 32:
            raise ValueError("Image axes must be positive multiples of 32")
        return width, height

    def _apply_seed(self, seed):
        seed = int(seed)
        if not 0 <= seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        self.seed = seed

    @property
    def identity(self):
        return {**self._checkpoint_identity, "size": self.size, "dtype": self.dtype,
                "seed": self.seed, "num_steps": self.num_steps, "cfg_scale": self.cfg_scale,
                "timestep_shift": self.timestep_shift if self.model_family == "sensenova_u15" else None,
                "attn_backend": self.attn_backend if self.model_family == "sensenova_u15" else "native_sdpa",
                "think": False, "sampler": "native_inference", "lora_precision": "exact_fp32_master"}

    def _load_weights(self):
        import torch
        path = self.checkpoint_path / "state.pt"
        if _digest(path) != self.complete["files_sha256"]["state.pt"]:
            raise ValueError("Selected FM checkpoint state.pt SHA256 mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (state.get("identity") != self.complete["identity"] or state.get("step") != self.complete["step"]
                or state.get("trainable_only") is not True):
            raise ValueError("FM checkpoint payload differs from its completion marker")
        weights = state.get("model")
        if not isinstance(weights, dict) or set(weights) != set(self.parameter_names):
            raise ValueError("FM checkpoint model keys differ from the full training parameter manifest")
        for name, value in weights.items():
            suffix = r"\.lora_[AB]\.default\.weight$" if self.model_family == "qwen_image21" else r"\.lora_[AB]$"
            if not re.search(suffix, name) or not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"Expected finite FP32 LoRA A/B weights only: {name}")
        return weights

    def _load(self):
        if self.client is not None:
            return
        import torch
        if str(_DRAW_ROOT) not in sys.path:
            sys.path.insert(0, str(_DRAW_ROOT))
        dtype = torch.bfloat16 if self.dtype in {"bf16", "bfloat16"} else torch.float32
        device = torch.device(self.device)
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        weights = self._load_weights()
        base = self._checkpoint_identity["base_model"]
        lora = self.config["finetune"].get("lora", {})
        if self.model_family == "qwen_image21":
            from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
            from peft import LoraConfig
            overrides = {}
            if self.config["model"].get("transformer_path"):
                overrides["transformer"] = QwenImage21Transformer2DModel.from_pretrained(
                    self.config["model"]["transformer_path"], torch_dtype=dtype, local_files_only=True)
            pipeline = QwenImage21Pipeline.from_pretrained(base, torch_dtype=dtype, local_files_only=True, **overrides)
            rank = int(lora.get("rank", lora.get("r", 16)))
            pipeline.transformer.add_adapter(LoraConfig(
                r=rank, lora_alpha=float(lora.get("alpha", lora.get("lora_alpha", rank))),
                lora_dropout=float(lora.get("dropout", lora.get("lora_dropout", 0.0))),
                target_modules=lora.get("target_modules") or ["to_k", "to_q", "to_v", "to_out.0"],
                init_lora_weights=False))
            # Place base first, then restore adapter FP32 masters: .to(dtype)
            # after restoration would silently round the selected checkpoint.
            pipeline.to(device=device, dtype=dtype)
            _restore_parameters(pipeline.transformer, weights)
            pipeline.text_encoder.requires_grad_(False).eval()
            pipeline.vae.requires_grad_(False).eval()
            self.client = pipeline
        else:
            from draw_align_sft.legacy_sensenova.model import load_source
            package, fingerprint = load_source(self.model_code_root)
            if fingerprint != self._checkpoint_identity["native_source_sha256"]:
                raise ValueError("SenseNova native inference code differs from the checkpoint's training source")
            package.set_attn_backend(self.attn_backend)
            loader = importlib.import_module("sensenova_u1.utils").load_model_and_tokenizer
            model, tokenizer = loader(base, dtype=dtype, device=device,
                                      device_map={"": str(device)} if device.type == "cuda" else None)
            _install_sensenova_lora(model, weights, lora)
            self.client, self.tokenizer = model, tokenizer

    def generate(self, prompt, destination, timeout):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Image generation requires a nonempty prompt")
        self._load()
        import torch
        from PIL import Image
        width, height = self._dimensions()
        with torch.inference_mode():
            if self.model_family == "qwen_image21":
                generator = torch.Generator(device=self.device).manual_seed(self.seed)
                result = self.client(prompt=prompt, width=width, height=height,
                    negative_prompt="" if self.cfg_scale > 1 else None,
                    true_cfg_scale=self.cfg_scale, num_inference_steps=self.num_steps,
                    num_images_per_prompt=1, generator=generator, output_type="pil", use_kv_cache=True)
                if len(result.images) != 1:
                    raise ValueError("Expected one Qwen-Image2.1 image")
                image = result.images[0]
                if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
                    rgba = image.convert("RGBA")
                    image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
                else:
                    image = image.convert("RGB")
            else:
                pixels = self.client.t2i_generate(self.tokenizer, prompt, image_size=(width, height),
                    cfg_scale=self.cfg_scale, cfg_norm="none", timestep_shift=self.timestep_shift,
                    cfg_interval=(0.0, 1.0), num_steps=self.num_steps, batch_size=1,
                    seed=self.seed, think_mode=False)
                if not isinstance(pixels, torch.Tensor) or pixels.shape != (1, 3, height, width):
                    raise ValueError("Unexpected native SenseNova output tensor shape")
                if not torch.isfinite(pixels).all():
                    raise ValueError("Native SenseNova generated nonfinite pixels")
                # Match native SenseNovaInferencer._tensor_to_pil exactly:
                # normalize/clamp in the sampler's dtype before promoting.
                values = ((pixels[0] * .5 + .5).clamp(0, 1).float().permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
                image = Image.fromarray(values, "RGB")
        if image.size != (width, height):
            raise ValueError("Native generator output dimensions differ from requested dimensions")
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, "PNG")
        self.last_metadata = {**self.identity, "usage": {}}
        return self.last_metadata
