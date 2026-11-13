"""Publish a checked HF checkpoint by merging the completed SFT LoRA on CPU.

No optimizer state is imported. Full verification checks every weight byte.
Published verification reuses an already published artifact after checking source
manifests, metadata and tensor structure; it deliberately trusts published weight
contents instead of re-reading the entire model. Failed staging folders cannot
be mistaken for a complete model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import uuid


PROCESSOR_FILES = (
    "chat_template.jinja", "chat_template.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json",
    "preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    total = path.stat().st_size
    consumed = 0
    last_report = time.monotonic()
    if total >= 100 * 2**20:
        print(f"Verifying checksum: {path.parent.name}/{path.name} ({total / 2**20:.0f} MiB)", flush=True)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
            consumed += len(block)
            if time.monotonic() - last_report >= 10:
                print(f"Checksum {path.name}: {consumed / 2**20:.0f}/{total / 2**20:.0f} MiB", flush=True)
                last_report = time.monotonic()
    return value.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _weight_header(path: Path) -> dict:
    """Validate safetensors structure and length without reading tensor payloads."""
    sizes = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
             "I32": 4, "U32": 4, "I64": 8, "U64": 8, "F16": 2,
             "BF16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1,
             "F8_E4M3FN": 1, "F8_E5M2FNUZ": 1, "F8_E4M3FNUZ": 1, "F8_E8M0": 1}
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        length = int.from_bytes(raw, "little")
        if len(raw) != 8 or not 2 <= length <= min(100 * 2**20, size - 8):
            raise ValueError("Invalid/truncated safetensors header: " + str(path))
        try:
            header = json.loads(stream.read(length))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid safetensors header: " + str(path)) from exc
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header: " + str(path))
    tensors = {name: item for name, item in header.items() if name != "__metadata__"}
    if not tensors:
        raise ValueError("Empty safetensors tensor inventory: " + str(path))
    intervals = []
    for name, item in tensors.items():
        if not isinstance(item, dict):
            raise ValueError("Invalid safetensors tensor entry: " + str(path))
        offsets, shape, dtype = item.get("data_offsets"), item.get("shape"), item.get("dtype")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(v) is not int or v < 0 for v in offsets)
                or offsets[1] < offsets[0] or not isinstance(shape, list)
                or any(type(v) is not int or v < 0 for v in shape) or dtype not in sizes
                or offsets[1] - offsets[0] != math.prod(shape) * sizes[dtype]):
            raise ValueError("Invalid safetensors tensor shape/offset: " + str(path))
        intervals.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("Overlapping or incomplete safetensors tensor data: " + str(path))
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError("Truncated or extra safetensors weight data: " + str(path))
    return tensors


def _check_weight_map(weight_map: dict, headers: dict) -> None:
    actual = {name: shard for shard, tensors in headers.items() for name in tensors}
    if (len(actual) != sum(len(tensors) for tensors in headers.values())
            or actual != weight_map):
        raise ValueError("Safetensors tensor inventory differs from checkpoint index")


def source_identity(base: Path, adapter: Path, *, published_identity: dict | None = None) -> dict:
    """Check original SFT identity, using full bytes or published weight metadata."""
    base, adapter = base.resolve(), adapter.resolve()
    complete = _json(adapter / "checkpoint_manifest.json")
    state = _json(adapter / "trainer_state.json")
    if complete.get("complete") is not True or complete.get("global_step") != state.get("global_step"):
        raise ValueError("SFT checkpoint is incomplete or its step differs from trainer_state")
    epoch = state.get("epoch")
    # A complete checkpoint manifest proves the save finished, not that the
    # requested training epochs finished. Retain legacy one-epoch admission,
    # while requiring explicit training completion for longer SFT runs.
    expected_epochs = complete.get("identity", {}).get("semantic_config", {}).get(
        "num_train_epochs", state.get("num_train_epochs", 1.0))
    if (isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or not math.isfinite(epoch)
            or isinstance(expected_epochs, bool) or not isinstance(expected_epochs, (int, float))
            or not math.isfinite(expected_epochs) or expected_epochs < 1
            or not math.isclose(epoch, expected_epochs, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("Expected the completed SFT checkpoint at its configured final epoch")
    max_steps = state.get("max_steps")
    if ((epoch != 1.0 or max_steps is not None)
            and (type(max_steps) is not int or max_steps <= 0 or state.get("global_step") != max_steps)):
        raise ValueError("Expected the completed SFT checkpoint at its final optimizer step")
    if _json(base / "config.json").get("model_type") not in {"qwen3_vl", "qwen3_5"}:
        raise ValueError("Expected a supported Qwen3-VL or Qwen3.5/V3 multimodal base model")
    if _json(adapter / "adapter_config.json").get("peft_type") != "LORA":
        raise ValueError("Expected a PEFT LoRA checkpoint")
    recorded_base = complete.get("identity", {}).get("model", {}).get("files_sha256", {})
    recorded_adapter = complete.get("files", {})
    index = _json(base / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map or any(not isinstance(name, str) for name in weight_map.values()):
        raise ValueError("Base checkpoint index has no valid weight shards")
    base_files = {"config.json", "model.safetensors.index.json", *weight_map.values()}
    base_files.update(name for name in PROCESSOR_FILES if (base / name).is_file())
    if (base / "generation_config.json").is_file():
        base_files.add("generation_config.json")
    adapter_files = {"adapter_config.json", "adapter_model.safetensors", "trainer_state.json"}
    adapter_files.update(name for name in PROCESSOR_FILES if (adapter / name).is_file())
    identity = {"base": {}, "adapter": {}, "checkpoint_step": state["global_step"], "sft_epoch": state["epoch"]}
    for folder, files, recorded, key in ((base, base_files, recorded_base, "base"),
                                         (adapter, adapter_files, recorded_adapter, "adapter")):
        headers = {}
        for name in sorted(files):
            path = folder / name
            if not path.resolve().is_relative_to(folder) or not path.is_file():
                raise ValueError("Missing or escaping checkpoint file: " + name)
            if published_identity is not None and name.endswith(".safetensors"):
                headers[name] = _weight_header(path)
                actual = recorded.get(name)
                if (not isinstance(actual, str) or len(actual) != 64
                        or actual != published_identity.get(key, {}).get(name)):
                    raise ValueError("Published source weight fingerprint differs from SFT manifest: " + name)
            else:
                actual = digest(path)
            # The SFT identity predates some optional HF index/generation files;
            # mandatory weights/config/tokenizer must have recorded fingerprints.
            mandatory = name.endswith(".safetensors") or name in {"config.json", "adapter_config.json", "tokenizer.json", "trainer_state.json"}
            if mandatory and name not in recorded:
                raise ValueError("Missing original SFT integrity fingerprint: " + name)
            if name in recorded and recorded[name] != actual:
                raise ValueError("SFT input checksum mismatch: " + name)
            identity[key][name] = actual
        if published_identity is not None and key == "base":
            _check_weight_map(weight_map, headers)
    identity["checkpoint_manifest_sha256"] = digest(adapter / "checkpoint_manifest.json")
    return identity


def validate_published(base: Path, adapter: Path, output: Path) -> dict:
    """Metadata/structure admission, not a replacement for a byte-integrity audit."""
    if not (output / "merge_manifest.json").is_file():
        raise ValueError("Published merged checkpoint missing; prepare it on CPU with --verification full first")
    complete = _json(output / "merge_manifest.json")
    if complete.get("status") != "complete" or not isinstance(complete.get("source_identity"), dict):
        raise ValueError("Published merged checkpoint is incomplete")
    if (complete.get("source_base") != str(base)
            or complete.get("source_adapter") != str(adapter)):
        raise ValueError("Published merged checkpoint has a different source path")
    identity = source_identity(base, adapter, published_identity=complete["source_identity"])
    if identity != complete["source_identity"]:
        raise ValueError("Published merged checkpoint has a different source identity")
    files = complete.get("files_sha256", {})
    actual_files = {str(p.relative_to(output)) for p in output.rglob("*")
                    if p.is_file() and p.name != "merge_manifest.json"}
    if (not isinstance(files, dict) or set(files) != actual_files
            or "config.json" not in files or "tokenizer.json" not in files
            or not any(name.endswith(".safetensors") for name in files)):
        raise ValueError("Published merged checkpoint file inventory is incomplete or changed")
    headers = {}
    for name, expected in files.items():
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError("Escaping published checkpoint file: " + name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Invalid published checksum: " + name)
        if name.endswith(".safetensors"):
            headers[name] = _weight_header(path)
        elif digest(path) != expected:
            raise ValueError("Published checkpoint metadata checksum mismatch: " + name)
    index = output / "model.safetensors.index.json"
    base_map = _json(base / "model.safetensors.index.json")["weight_map"]
    if index.is_file():
        weight_map = _json(index).get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("Published checkpoint has no valid tensor index")
    elif set(headers) == {"model.safetensors"}:
        weight_map = {name: "model.safetensors" for name in base_map}
    else:
        raise ValueError("Published sharded checkpoint is missing its tensor index")
    _check_weight_map(weight_map, headers)
    # The published index, whose hash was checked above, is authoritative.
    # HF may omit unsupported auxiliary heads from a base checkpoint (e.g.
    # Qwen3.5 MTP tensors) when it saves the merged inference/training model.
    return complete


def validate_output(output: Path, identity: dict) -> dict:
    complete = _json(output / "merge_manifest.json")
    if complete.get("status") != "complete" or complete.get("source_identity") != identity:
        raise ValueError("Existing merge output has a different or incomplete source identity")
    files = complete.get("files_sha256", {})
    actual_files = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file() and path.name != "merge_manifest.json"}
    if set(files) != actual_files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Merged output file inventory differs from its complete manifest")
    for name, expected in files.items():
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()) or digest(path) != expected:
            raise ValueError("Merged output checksum mismatch: " + name)
    return complete


@contextmanager
def output_lock(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.name + ".merge.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is merging this SFT checkpoint") from None
        yield


def merge(base: Path, adapter: Path, output: Path, *, verification: str = "full") -> dict:
    base, adapter, output = base.resolve(), adapter.resolve(), output.resolve()
    if output.is_relative_to(base) or output.is_relative_to(adapter) or base.is_relative_to(output) or adapter.is_relative_to(output):
        raise ValueError("Merge destination must be separate from the source checkpoint directories")
    if verification not in {"published", "full"}:
        raise ValueError("Unknown checkpoint verification mode: " + verification)
    with output_lock(output):
        if verification == "published":
            print("Checking published SFT metadata and safetensors structure; trusting published weight contents (no full weight checksums).", flush=True)
            manifest = validate_published(base, adapter, output)
            print("Published merged SFT checkpoint ready: " + str(output), flush=True)
            return manifest
        print("Verifying original SFT and base checkpoint checksums...", flush=True)
        identity = source_identity(base, adapter)
        if output.exists():
            manifest = validate_output(output, identity)
            print("Verified existing merged SFT checkpoint: " + str(output), flush=True)
            return manifest
        import torch
        from importlib.metadata import version
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Small low-rank products slow down badly when a large CPU host launches
        # hundreds of BLAS workers per layer.
        torch.set_num_threads(min(torch.get_num_threads(), 8))

        staging = output.with_name("." + output.name + ".staging-" + uuid.uuid4().hex)
        staging.mkdir()
        print("Merging SFT LoRA into the verified Qwen multimodal base on CPU...", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            str(base), dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
            local_files_only=True, attn_implementation="eager",
        )
        peft_model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True)
        merged = peft_model.merge_and_unload(safe_merge=True)
        if any("lora_" in name for name, _ in merged.named_parameters()):
            raise RuntimeError("LoRA parameters remain after merge_and_unload")
        merged.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
        # Use exact SFT processor/tokenizer files rather than a new library's
        # serialization. Fall back to base only for files not exported by SFT.
        for name in PROCESSOR_FILES:
            source = adapter / name if (adapter / name).is_file() else base / name
            if source.is_file():
                shutil.copyfile(source, staging / name)
        AutoProcessor.from_pretrained(staging, local_files_only=True)
        files = {str(path.relative_to(staging)): digest(path) for path in sorted(staging.rglob("*")) if path.is_file()}
        manifest = {
            "schema_version": 1, "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
            "source_base": str(base), "source_adapter": str(adapter), "source_identity": identity,
            "merge_method": "peft.merge_and_unload(safe_merge=True)", "dtype": "bfloat16", "device": "cpu",
            "versions": {name: version(name) for name in ("torch", "transformers", "peft", "safetensors")},
            "files_sha256": files,
        }
        (staging / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
        print("Merged SFT checkpoint ready: " + str(output), flush=True)
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification", choices=("published", "full"), default="full",
                        help="published: reuse completed artifact with metadata/header checks; full: verify all weight bytes and prepare if missing")
    args = parser.parse_args(argv)
    merge(args.base, args.adapter, args.output, verification=args.verification)


if __name__ == "__main__":
    main()
