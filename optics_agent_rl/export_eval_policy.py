"""Resolve a published GRPO adapter against its original SFT-merged base.

This is a read-only, CPU-only preflight. It validates the small adapter and its
native source shards; it does not rehash or allocate the entire 27B base model.
The exporter already produces PEFT weights, so no second conversion is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _weight_header(path):
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        _require(len(prefix) == 8, "Invalid adapter safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        _require(0 < size < 16 * 1024 * 1024, "Invalid adapter safetensors header length")
        encoded = stream.read(size)
    _require(len(encoded) == size, "Truncated adapter safetensors header")
    header = json.loads(encoded)
    _require(isinstance(header, dict), "Adapter header must be an object")
    return {key: value for key, value in header.items() if key != "__metadata__"}


def resolve_eval_policy(checkpoint, *, adapter=None, verify_hashes=True):
    """Return exact evaluation base/adapter paths or reject incomplete provenance.

    ``verify_hashes=False`` skips only the ~200MB native/adapter payload hashes;
    small metadata identities remain checked. Use the default for real launches.
    """
    checkpoint = Path(checkpoint).resolve()
    match = re.fullmatch(r"global_step_([1-9][0-9]*)", checkpoint.name)
    _require(match is not None and checkpoint.parent.name == "checkpoints",
             "Expected a native checkpoints/global_step_N directory")
    step = int(match.group(1))
    run = checkpoint.parent.parent
    published = Path(adapter).absolute() if adapter is not None else run / "exports" / f"step_{step}"
    _require(published.is_dir(), f"Missing completed GRPO adapter export: {published}")
    manifest = _json(published / "export_manifest.json")
    state = _json(published / "export_state.json")
    _require(manifest.get("status") == state.get("status") == "complete"
             and manifest.get("step") == state.get("step") == step,
             "GRPO export is incomplete or belongs to another step")
    _require(Path(manifest.get("checkpoint", "")).resolve() == checkpoint
             and Path(manifest.get("run_dir", "")).resolve() == run,
             "GRPO export belongs to another source checkpoint/run")
    _require(manifest.get("tensor_round_trip_exact") is True,
             "GRPO export lacks exact tensor round-trip verification")
    latest = int((run / "checkpoints" / "latest_checkpointed_iteration.txt").read_text().strip())
    _require(latest >= step, "Requested checkpoint is not covered by the completed save marker")

    identity_path = run / "run_identity.json"
    _require(_digest(identity_path) == manifest.get("training_identity_sha256"),
             "GRPO training identity changed after adapter export")
    identity = _json(identity_path)
    model = identity["actor_rollout_ref"]["model"]
    base = Path(manifest.get("required_base_model", "")).resolve()
    _require(Path(model["path"]).resolve() == base, "GRPO base differs from training identity")
    merge_path = base / "merge_manifest.json"
    merge_sha = _digest(merge_path)
    _require(merge_sha == manifest.get("base_merge_manifest_sha256")
             == identity.get("source_manifests", {}).get("merge_manifest.json"),
             "Required SFT-merged base identity differs from the GRPO training base")
    merge = _json(merge_path)
    _require(merge.get("status") == "complete", "Required SFT base merge is incomplete")
    for key in ("source_base", "source_adapter", "created_at"):
        _require(manifest.get("base_identity", {}).get(key) == merge.get(key),
                 "GRPO export has inconsistent SFT base provenance")
    base_files = merge.get("files_sha256", {})
    for name in ("config.json", "model.safetensors.index.json"):
        _require(_digest(base / name) == base_files.get(name), f"Required base metadata changed: {name}")
    _require(_json(base / "config.json").get("model_type") in {"qwen3_5", "qwen3_vl"},
             "Unsupported multimodal GRPO policy base")
    weight_map = _json(base / "model.safetensors.index.json").get("weight_map", {})
    _require(isinstance(weight_map, dict) and weight_map, "Required base weight index is empty")
    for name in set(weight_map.values()):
        path = base / name
        _require(path.resolve().is_relative_to(base) and name in base_files
                 and path.is_file() and path.stat().st_size > 0,
                 f"Missing required SFT base weight shard: {name}")

    actor = checkpoint / "actor"
    tokenizer = actor / "huggingface"
    for name in ("tokenizer.json", "chat_template.jinja", "processor_config.json", "generation_config.json"):
        _require(_digest(tokenizer / name) == base_files.get(name),
                 f"GRPO processor/tokenizer differs from the training-pinned base: {name}")
    _require(_digest(base / "tokenizer_config.json") == base_files.get("tokenizer_config.json"),
             "Required base tokenizer configuration changed")
    base_tokenizer, trained_tokenizer = _json(base / "tokenizer_config.json"), _json(tokenizer / "tokenizer_config.json")
    # HF serialization may make these defaults explicit; they do not change
    # token IDs, added tokens or the conversation template.
    for value in (base_tokenizer, trained_tokenizer):
        value.pop("local_files_only", None)
        value.setdefault("padding_side", "right")
        value.setdefault("pad_token_type_id", 0)
        value.setdefault("max_length", None)
        value.setdefault("pad_to_multiple_of", None)
    _require(base_tokenizer == trained_tokenizer,
             "GRPO tokenizer configuration differs semantically from the SFT base")
    fsdp = _json(actor / "fsdp_config.json")
    world = fsdp.get("world_size")
    _require(fsdp.get("FSDP_version") == 2 and type(world) is int and world > 0,
             "Expected a complete FSDP2 checkpoint")
    expected_shards = {f"model_world_size_{world}_rank_{rank}.pt" for rank in range(world)}
    shard_hashes = manifest.get("source_model_shards_sha256", {})
    _require(set(shard_hashes) == expected_shards, "GRPO export does not cover every source model rank")
    for name in sorted(expected_shards):
        path = actor / name
        _require(path.is_file() and path.stat().st_size > 0, f"Missing GRPO source shard: {name}")
        if verify_hashes:
            _require(_digest(path) == shard_hashes[name], f"GRPO source shard checksum changed: {name}")
    exports = manifest.get("files_sha256", {})
    _require(set(exports) == {"adapter_config.json", "adapter_model.safetensors"},
             "GRPO adapter file inventory is incomplete")
    for name, expected in exports.items():
        path = published / name
        _require(path.is_file() and path.stat().st_size > 0, f"Missing GRPO adapter file: {name}")
        if verify_hashes or name.endswith(".json"):
            _require(_digest(path) == expected, f"GRPO adapter checksum changed: {name}")
    config = _json(published / "adapter_config.json")
    meta = _json(actor / "lora_train_meta.json")
    nested = model.get("lora") or {}
    if int(nested.get("rank", 0)) > 0:
        rank, alpha, targets = nested["rank"], nested["alpha"], nested["target_modules"]
    else:
        rank, alpha, targets = model["lora_rank"], model["lora_alpha"], model["target_modules"]
    _require(rank > 0 and alpha > 0
             and config.get("r") == meta.get("r") == manifest.get("lora_rank") == rank
             and config.get("lora_alpha") == meta.get("lora_alpha") == manifest.get("lora_alpha") == alpha,
             "GRPO LoRA rank/alpha differ from training")
    _require(config.get("peft_type") == "LORA" and config.get("task_type") == meta.get("task_type") == "CAUSAL_LM",
             "Expected a causal-LM GRPO LoRA adapter")
    _require(Path(config.get("base_model_name_or_path", "")).resolve() == base,
             "GRPO adapter must load on its SFT-merged base, not raw Qwen or a second SFT adapter")
    _require(set(config.get("target_modules", [])) == set(manifest.get("target_modules", [])) == set(targets),
             "GRPO LoRA target modules differ from training")
    _require(not config.get("use_dora") and not config.get("use_rslora")
             and not config.get("modules_to_save") and config.get("bias") == "none",
             "Unsupported extra GRPO adapter state")
    header = _weight_header(published / "adapter_model.safetensors")
    pairs = {}
    parameters = 0
    tensor_dtypes = set()
    for name, tensor in header.items():
        key = re.fullmatch(r"(.+)\.lora_([AB])\.weight", name)
        _require(key is not None, "GRPO adapter contains non-LoRA parameters")
        shape = tensor.get("shape", [])
        _require(len(shape) == 2 and all(type(x) is int and x > 0 for x in shape)
                 and tensor.get("dtype") in {"BF16", "F32"}, "Invalid GRPO adapter tensor shape or dtype")
        tensor_dtypes.add({"BF16": "torch.bfloat16", "F32": "torch.float32"}[tensor["dtype"]])
        module, side = key.groups()
        _require(module.rsplit(".", 1)[-1] in targets and shape[0 if side == "A" else 1] == rank,
                 "GRPO adapter tensor differs from target module/rank contract")
        pairs.setdefault(module, set()).add(side)
        parameters += math.prod(shape)
    _require(header and len(header) == manifest.get("tensor_count")
             and parameters == manifest.get("parameter_count")
             and all(sides == {"A", "B"} for sides in pairs.values()),
             "GRPO adapter is missing LoRA A/B tensors")
    # Newer GRPO runs retain FP32 adapter masters. Preserve the native export
    # rather than rounding it to satisfy the older BF16-only reader. Old BF16
    # manifests without an explicit dtype inventory remain compatible.
    declared_dtypes = manifest.get("tensor_dtypes", ["torch.bfloat16"])
    _require(isinstance(declared_dtypes, list) and set(declared_dtypes) == tensor_dtypes,
             "GRPO adapter dtype differs from its exact native export manifest")
    return {
        "status": "verified", "global_step": step,
        "checkpoint": str(checkpoint), "base_model": str(base),
        "adapter_path": str(published), "adapter_resolved_path": str(published.resolve()),
        "tokenizer_path": str(tokenizer),
        "export_manifest_sha256": _digest(published / "export_manifest.json"),
        "training_identity_sha256": manifest["training_identity_sha256"],
        "base_merge_manifest_sha256": merge_sha,
        "sft_source_adapter": merge.get("source_adapter"),
        "lora_rank": rank, "lora_alpha": alpha,
        "tensor_count": len(header), "parameter_count": parameters,
        "tensor_dtypes": sorted(tensor_dtypes),
        "payload_hashes_verified": bool(verify_hashes), "base_weights_rehashed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(resolve_eval_policy(args.checkpoint, adapter=args.adapter), indent=2))


if __name__ == "__main__":
    main()
