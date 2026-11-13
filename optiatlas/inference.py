"""Relocatable exported LoRA loading with the original native sampler.

Full training archives are not needed for inference. Weight identity is checked
against the selected published adapter SHA, before any GPU model is loaded.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

FINAL_GENERATOR_SHA = "9ecae447a4c580742d1f33465ec6596a6a5d16eacaca77936905043e1580796f"
RL95_SHA = "0833c06b914a0f14c7b13025e9622dd72e42aead95683e5f2aa0653d722d9838"
SFT_MERGE_SHA = "e9aaec367ddb5139c459bdbeb7d21cff0c9ef6b3b34a83448fa66ed47efae5a1"
POLICIES = {
    "rl95": ("OptiAtlas-RL95", "policy_rl_adapter", "policy_tokenizer", RL95_SHA),
    "binary": ("OptiAtlas-Binary95", "policy_binary_adapter", "policy_binary_tokenizer", "7a3017df17929fe210b14a74940955b01e14df3d0e28cd7e3e237ae1fb735767"),
    "image": ("OptiAtlas-Image95", "policy_image_adapter", "policy_image_tokenizer", "aa936f5f22402db557de3633d8342ddb39b006e478b73289f82eb31f19e2bd33"),
}


def validate_sft_base(base):
    """Check the published merge identity without rereading 55 GB of tensors."""
    base=Path(base)
    published=base.parent.parent/"weights_manifest.json"
    local=base/"merge_manifest.json"
    if published.is_file():
        manifest=json.loads(published.read_text())
        if manifest.get("policy_sft_source_merge_manifest_sha256")==SFT_MERGE_SHA:
            entries={key.removeprefix("policy/sft_base/"):value for key,value in manifest["files"].items() if key.startswith("policy/sft_base/")}
            for name,entry in entries.items():
                path=base/name
                if not path.is_file() or path.stat().st_size!=entry["bytes"]:
                    raise ValueError("Published SFT base file missing or wrong size: "+name)
            for name in ("config.json","model.safetensors.index.json"):
                if sha256(base/name)!=entries[name]["sha256"]:
                    raise ValueError("Published SFT base metadata changed: "+name)
            return
    if local.is_file():
        manifest=json.loads(local.read_text());identity=manifest.get("source_identity",{})
        if identity.get("checkpoint_step")==309 and identity.get("sft_epoch")==3.0:
            for name in ("config.json","model.safetensors.index.json"):
                if sha256(base/name)!=manifest.get("files_sha256",{}).get(name):
                    raise ValueError("Local SFT merge metadata changed: "+name)
            return
    raise ValueError("The policy base must be the verified SFT309 three-epoch merge; see the published weights manifest")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def exported_identity(image):
    base = Path(image["model_path"]).expanduser().resolve()
    checkpoint = image.get("checkpoint_path")
    if not base.is_dir():
        raise ValueError("generator_base must be an existing Qwen-Image-2.1 directory")
    identity = {"model_family": "qwen_image21", "base_model": str(base), "checkpoint_kind": "base"}
    if checkpoint:
        directory = Path(checkpoint).expanduser().resolve()
        meta = json.loads((directory / "draw_align_export.json").read_text())
        peft = json.loads((directory / "peft_config/adapter_config.json").read_text())
        expected = image.get("state_sha256")
        if not expected or len(expected) != 64:
            raise ValueError("Every adapted generator requires its published state_sha256")
        actual = sha256(directory / "pytorch_lora_weights.safetensors")
        if actual != expected:
            raise ValueError("Selected generator adapter SHA256 differs from its published identity")
        if meta.get("model_type") != "qwen_image21" or meta.get("format") != "diffusers_lora":
            raise ValueError("Expected the native Qwen-Image-2.1 export")
        if meta.get("initial_lora_path") or meta.get("initial_transformer_path"):
            raise ValueError("This release requires original-base independently trained adapters")
        if peft.get("r") != 32 or peft.get("lora_alpha") != 64:
            raise ValueError("Published generator adapters require r32 / alpha64")
        identity.update(checkpoint_kind="exported_lora", checkpoint_path=str(directory), state_sha256=actual,
                        adapter_semantics="complete_updated_adapter_not_additive_delta")
    return identity


def install_release_provider():
    """Replace only deployment lookup, preserving the benchmark and sampler."""
    from optics_experiments.fm_image_provider import LocalFMCheckpointProvider
    from optics_experiments import image_provider, benchmark_generate

    class ExportedProvider(LocalFMCheckpointProvider):
        def __init__(self, **image):
            self._checkpoint_identity = exported_identity(image)
            self.checkpoint_path = Path(image["checkpoint_path"]) if image.get("checkpoint_path") else None
            self.model_family = "qwen_image21"
            self.config = {"model": {}}
            self._configure_inference(image.get("device", "cuda:0"), image.get("dtype", "bf16"),
                image.get("size", "1024x1024"), image.get("seed", 42), image.get("num_steps", 40),
                image.get("cfg_scale", 4.0), 3.0, None, "sdpa")

        def _load(self):
            if self.client is not None:
                return
            import torch
            from diffusers import QwenImage21Pipeline
            from draw_align_sft.models.lora_precision import load_qwen_lora_exact
            pipe = QwenImage21Pipeline.from_pretrained(self._checkpoint_identity["base_model"],
                torch_dtype=torch.bfloat16, local_files_only=True)
            pipe.to(device=self.device, dtype=torch.bfloat16)
            if self.checkpoint_path:
                load_qwen_lora_exact(pipe, self.checkpoint_path)
            for component in (pipe.transformer, pipe.text_encoder, pipe.vae):
                component.requires_grad_(False).eval()
            self.client = pipe

    original_factory = image_provider.make_image_provider
    def make_provider(cfg):
        if cfg.get("provider") == "draw_align_checkpoint" and cfg.get("release_export", False):
            return ExportedProvider(**{k:v for k,v in cfg.items() if k not in {"provider", "release_export"}})
        return original_factory(cfg)
    image_provider.make_image_provider = make_provider
    # The benchmark binds a complete training checkpoint in _spec. Release
    # exports instead bind their bytes plus model identity, without requiring
    # optimizer/EMA archives or rewriting their provenance metadata.
    original_spec = benchmark_generate._spec
    def spec(values, records):
        from copy import deepcopy
        if not values["image"].get("release_export", False):
            return original_spec(values, records)
        current = deepcopy(values)
        current["image"]["provider"] = "release_export"
        result = original_spec(current, records)
        from optics_experiments.logging import fingerprint
        result["identity"]["finetuned_checkpoint"] = exported_identity(values["image"])
        result["identity"]["release_loader_sha256"] = sha256(Path(__file__))
        result["execution_fingerprint"] = fingerprint(result["identity"])
        return result
    benchmark_generate._spec = spec
    return ExportedProvider
