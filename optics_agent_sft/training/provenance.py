"""Training-data admission and reproducible model/data provenance.

This module performs only filesystem/JSON work. Its hashes deliberately replace
machine-local image paths with image bytes, so copied datasets compare equally
across nodes. It never imports a processor, model, or GPU dependency.
"""
from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from ..data import Admission, admit_trajectories, load_dataset, read_json, sha256_file
from .architectures import SUPPORTED_MODEL_TYPES
from .config import canonical_hash
from .distributed import Topology

def dataset_identity(admission: Admission) -> dict[str, Any]:
    """Bind ordered decisions, visual bytes and human approvals, never mount paths."""
    samples = []
    for original in admission.samples:
        sample = copy.deepcopy(original)
        hashes = iter(sample["image_sha256"])
        count = 0
        for message in sample["messages"]:
            if not isinstance(message.get("content"), list):
                continue
            for part in message["content"]:
                if part.get("type") == "image_url":
                    part["image_url"] = {"url": "sha256:" + next(hashes)}
                    count += 1
        if count != len(sample["image_sha256"]):
            raise ValueError("Sample image hashes and image occurrences disagree")
        samples.append(sample)
    sources = [{key: row[key] for key in ("trajectory_sha256", "human_review_sha256", "selected_step_ids", "scene_family", "split")}
               for row in admission.accepted]
    # Preserve historical human-only identities; machine-curated data must also
    # bind the exact automatic judge evidence used for its admission.
    for source, accepted in zip(sources, admission.accepted):
        if accepted.get("review_type") == "model":
            source.update(review_type="model", machine_review_sha256=accepted["machine_review_sha256"])
    return {"schema_version": 1, "ordered_samples_sha256": canonical_hash(samples),
            "source_approvals_sha256": canonical_hash(sources), "decision_samples": len(samples),
            "split_counts": {split: sum(s["split"] == split for s in samples) for split in ("train", "validation", "demo")}}



def model_identity(directory: str | Path, *, hash_workers: int = 1) -> dict[str, Any]:
    """Hash complete local model/tokenizer/processor bytes, including indexed shards."""
    if type(hash_workers) is not int or not 1 <= hash_workers <= 32:
        raise ValueError("model hash_workers must be an integer in [1, 32]")
    directory = Path(directory)
    if not (directory / "config.json").is_file():
        raise ValueError("Local model snapshot is missing config.json")
    if read_json(directory / "config.json").get("model_type") not in SUPPORTED_MODEL_TYPES:
        raise ValueError("Unsupported multimodal policy architecture; expected a verified Qwen3-VL or Qwen3.5 family")
    suffixes = {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken", ".jinja"}
    paths = sorted(directory.rglob("*"))
    if any(p.is_symlink() and p.is_dir() for p in paths):
        raise ValueError("Model snapshot directory symlinks are unsupported; ordinary HF snapshot file symlinks are allowed")
    paths = [p for p in paths if p.is_file()
             and not any(part.startswith(".") for part in p.relative_to(directory).parts)
             and p.suffix in suffixes]

    def state(path):
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    # Hash every byte on every launch. Overlap independent shard reads on the
    # node leader; this never substitutes a persisted size/mtime cache for SHA.
    before = {p: state(p) for p in paths}

    def digest(path):
        value = sha256_file(path)
        if state(path) != before[path]:
            raise ValueError("Model snapshot changed while computing its identity")
        return str(path.relative_to(directory)), value

    if hash_workers == 1:
        files = dict(map(digest, paths))
    else:
        with ThreadPoolExecutor(max_workers=hash_workers, thread_name_prefix="sft-model-sha") as pool:
            files = dict(pool.map(digest, paths))
    if any(state(p) != before[p] for p in paths):
        raise ValueError("Model snapshot changed while computing its identity")
    for name in files:
        if name.endswith((".safetensors.index.json", ".bin.index.json")):
            weight_map = read_json(directory / name).get("weight_map", {})
            if not weight_map or any(shard not in files for shard in weight_map.values()):
                raise ValueError("Every indexed model weight shard must be included in the snapshot hash inventory")
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("Local model snapshot has no model weights; processor-only directories cannot train")
    return {"files_sha256": files, "snapshot_sha256": canonical_hash(files)}



def admit_training_data(config: dict[str, Any], *, dry_run: bool, allow_demo: bool) -> tuple[Admission, list[dict], list[dict]]:
    """Revalidate curated sources, then separate train/validation decisions.

    Demo rows can be inspected by dry-run only; they are never optimizer input.
    """
    if allow_demo and not dry_run:
        raise ValueError("Demo trajectories are only allowed in dry-run, never training")
    admission = (load_dataset(config["dataset_dir"], reserved_paths=config["reserved_manifests"], allow_demo=allow_demo,
                              validation_workers=config.get("dataset_validation_workers", 1))
                 if config.get("dataset_dir") else
                 admit_trajectories(config["runs"], reserved_paths=config["reserved_manifests"], allow_demo=allow_demo))
    if not admission.samples:
        raise ValueError("No admitted decision samples: " + json.dumps(admission.summary(), ensure_ascii=False))
    training = [s for s in admission.samples if s["split"] == "train" or (dry_run and allow_demo and s["is_demo"])]
    validation = [s for s in admission.samples if s["split"] == "validation"]
    if not training:
        raise ValueError("No admitted train split samples")
    if not dry_run and any(s["is_demo"] for s in admission.samples):
        raise ValueError("Demo samples are never trainable")
    return admission, training, validation



def initial_manifest(config: dict[str, Any], admission: Admission, *, dry_run: bool, topology: Topology) -> dict[str, Any]:
    """Describe planned data/loss/topology before any model is constructed."""
    world = config["expected_world_size"] if dry_run else topology.world_size
    return {"schema_version": 2, "package_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "validated_only",
            "dry_run": dry_run, "config": config, "data": admission.summary(),
            "data_identity": dataset_identity(admission),
            "distributed": {"backend": config["distributed_backend"], "world_size": world,
                            "global_batch_size": config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] * world,
                            "gradient_normalization": "global supervised tokens across each accumulation window",
                            "shared_output_required": world > 1},
            "loss_scope": "one native assistant message/tool_calls and turn terminator only",
            "evaluation_loss_scope": "macro mean of per-decision supervised-token cross entropy; eval batch size is one",
            "message_normalization": "native-tool-text-image-user-v1",
            "limitations": ["Exact hashes cannot detect paraphrases; scene families require human curation.",
                            "CPU dry-run does not load or validate a real processor or perform optimization.",
                            "DDP/ZeRO-2 GPU and multi-node execution require validation on the target cluster."]}
