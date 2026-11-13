"""Resolved training model, native source, reward and runtime identities.

Only inspect already loaded objects and local bytes. This module never downloads
models. Call on rank zero and broadcast its result before constructing run IDs.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from functools import partial
from pathlib import Path
import platform
import re
import time


def _file_hash(path, *, progress=None):
    digest = hashlib.sha256()
    read_bytes, last_report = 0, time.monotonic()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            read_bytes += len(block)
            if progress is not None and time.monotonic() - last_report >= 5:
                progress(read_bytes)
                last_report = time.monotonic()
    if progress is not None:
        progress(read_bytes)
    return digest.hexdigest()


def _local_identity(path, *, source_only=False, progress=None):
    path = Path(path).expanduser().resolve()
    suffixes = {".json", ".safetensors", ".bin", ".pt", ".pth", ".model", ".txt", ".tiktoken", ".vocab", ".merges", ".py"}
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()
                                                               and (p.suffix == ".py" if source_only else p.suffix in suffixes))
    if not paths:
        raise ValueError(f"No identity-bearing files in local {'source' if source_only else 'model'} directory")
    if not source_only and not any(p.suffix in {".safetensors", ".bin", ".pt", ".pth"} for p in paths):
        raise ValueError("Local model identity requires checkpoint weight files")
    total = sum(p.stat().st_size for p in paths)
    completed, files = 0, {}
    if progress is not None:
        progress(component=str(path), bytes_read=0, total_bytes=total, files_completed=0, files_total=len(paths))
    for index, p in enumerate(paths):
        relative = p.name if path.is_file() else str(p.relative_to(path))
        def report(read_bytes):
            progress(component=str(path), file=relative, bytes_read=completed + read_bytes,
                     total_bytes=total, files_completed=index, files_total=len(paths))
        size = p.stat().st_size
        callback = report if progress is not None and size >= 64 * 1024 * 1024 else None
        files[relative] = _file_hash(p, progress=callback)
        completed += size
    if progress is not None:
        progress(component=str(path), bytes_read=completed, total_bytes=total,
                 files_completed=len(paths), files_total=len(paths))
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"kind": "local_bytes", "path": str(path), "sha256": digest, "files_sha256": files}


def _field(value, name):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _commit(value):
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40,64}", value) else None


def _snapshot(path):
    if not isinstance(path, (str, Path)):
        return None
    parts = Path(path).parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts) and _commit(parts[index + 1]):
            return parts[index + 1]
    return None


def _resolved_components(owner):
    components = {"root": owner}
    for key in ("transformer", "vae", "text_encoder", "tokenizer", "processor", "scheduler", "model"):
        value = getattr(owner, key, None)
        if value is not None:
            components[key] = value
    processor = getattr(owner, "processor", None)
    if getattr(processor, "tokenizer", None) is not None:
        components["processor.tokenizer"] = processor.tokenizer
    result = {}
    for name, component in components.items():
        config = getattr(component, "config", None)
        kwargs = getattr(component, "init_kwargs", None)
        for candidate in (component, config, kwargs):
            commit = _commit(_field(candidate, "_commit_hash"))
            location = _field(candidate, "_name_or_path") or _field(candidate, "name_or_path")
            commit = commit or _snapshot(location)
            if commit:
                result[name] = commit
                break
    return result


def _explicit_identity(owner):
    metadata = getattr(owner, "provenance_metadata", None)
    if metadata is None:
        return None
    if not isinstance(metadata, dict) or not metadata.get("identity"):
        raise ValueError("Injected component provenance_metadata requires a nonempty identity")
    # Ensure the supplied metadata can participate in stable JSON run identities.
    return {"kind": "explicit_injected", "metadata": json.loads(json.dumps(metadata, sort_keys=True))}


def _model_identity(source, revision, owner, *, progress=None):
    if source and Path(str(source)).expanduser().exists():
        return _local_identity(source, progress=progress)
    explicit = _explicit_identity(owner)
    if explicit is not None:
        return explicit
    if not source:
        raise ValueError("Model provenance requires a local checkpoint, resolved HF repository or explicit injected identity")
    resolved = _resolved_components(owner)
    pinned = _commit(revision)
    if pinned:
        resolved.setdefault("requested_repository", pinned)
    # Diffusers component configs may not retain _commit_hash. Resolve the
    # already cached model_index.json pointer without issuing network requests.
    # Prefer commits retained by the loaded objects over a mutable cached ref.
    if not resolved:
        from huggingface_hub import try_to_load_from_cache
        for filename in ("model_index.json", "config.json"):
            cached = try_to_load_from_cache(str(source), filename, revision=revision)
            snapshot = _snapshot(cached)
            if snapshot:
                resolved["cached_repository"] = snapshot
                break
    if not resolved:
        raise ValueError("Cannot establish loaded HF model revision; use a pinned commit or a local checkpoint directory")
    return {"kind": "huggingface_revision", "repository": str(source), "requested_revision": revision,
            "resolved_revisions": resolved}


def _custom_reward_identity(reward, *, progress=None):
    function = reward
    arguments, keywords = [], {}
    while isinstance(function, partial):
        arguments = list(function.args) + arguments
        keywords = {**(function.keywords or {}), **keywords}
        function = function.func
    owner = function if inspect.isfunction(function) or inspect.ismethod(function) else type(function)
    filename = inspect.getsourcefile(owner) or inspect.getfile(owner)
    if not filename or not Path(filename).is_file():
        raise ValueError("Custom reward needs an inspectable source module or explicit provenance_metadata")
    metadata = getattr(reward, "provenance_metadata", getattr(function, "provenance_metadata", None))
    declared_paths = metadata.get("model_paths", []) if isinstance(metadata, dict) else []
    if isinstance(declared_paths, (str, Path)):
        declared_paths = [declared_paths]
    return {"kind": "custom_python_hook", "module": owner.__module__, "qualname": owner.__qualname__,
            "source_path": str(Path(filename).resolve()), "source_sha256": _file_hash(filename),
            "bound_arguments": json.loads(json.dumps(arguments)),
            "bound_keywords": json.loads(json.dumps(keywords, sort_keys=True)),
            "declared_model_weights": [_local_identity(path, progress=progress) for path in declared_paths],
            "external_model_weights": "declared_paths_hashed" if declared_paths else "not_declared_by_hook",
            "metadata": metadata}


def training_provenance(config, adapter, reward=None, *, progress=None):
    model_config = config["model"]
    source = model_config.get("pretrained_model_name_or_path", model_config.get("model_path"))
    owner = getattr(adapter, "pipeline", None) or getattr(adapter, "pipe", None) or adapter
    # Injection metadata belongs to the adapter, even when it wraps a pipeline.
    if _explicit_identity(adapter) is not None and not (source and Path(str(source)).expanduser().exists()):
        model = _explicit_identity(adapter)
    else:
        model = _model_identity(source, model_config.get("revision"), owner, progress=progress)
    warm_starts = {}
    for key in ("transformer_path", "lora_path"):
        if model_config.get(key):
            path = Path(model_config[key]).expanduser()
            if not path.exists():
                raise ValueError(f"{key} provenance requires the resolved local warm-start checkpoint")
            warm_starts[key] = _local_identity(path, progress=progress)
    native_source = None
    if model_config.get("model_code_root"):
        native_source = _local_identity(model_config["model_code_root"], source_only=True, progress=progress)
        loaded = getattr(getattr(adapter, "native", None), "source_fingerprint", None)
        if loaded:
            # Match legacy_sensenova.model.load_source's fingerprint algorithm
            # and detect source edits between importing and recording the run.
            digest = hashlib.sha256()
            for relative in sorted(Path(name) for name in native_source["files_sha256"]):
                name = str(relative)
                value = native_source["files_sha256"][name]
                digest.update(name.encode())
                digest.update(value.encode())
            if digest.hexdigest() != loaded:
                raise ValueError("SenseNova native source changed after the model was loaded")
            native_source["loaded_sha256"] = loaded
    elif model_config.get("name") == "sensenova_u15" and model["kind"] != "explicit_injected":
        raise ValueError("SenseNova provenance requires its native model_code_root")
    reward_identity = None
    if reward is not None:
        reward_config = config.get("reward", {})
        reward_source = reward_config.get("model_name", reward_config.get("model_path"))
        explicit = _explicit_identity(reward)
        if reward_config.get("kind") == "custom" or (not reward_source and explicit is None):
            reward_model = _custom_reward_identity(reward, progress=progress)
        else:
            reward_model = explicit or _model_identity(reward_source, reward_config.get("revision"), reward, progress=progress)
        reward_identity = {"model": reward_model,
                           "metadata": getattr(reward, "metadata", None)}
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "diffusers", "peft", "Pillow", "safetensors", "huggingface_hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"format": 1, "model": model, "warm_starts": warm_starts, "native_source": native_source,
            "reward": reward_identity, "runtime_versions": versions}
