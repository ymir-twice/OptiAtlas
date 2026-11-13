"""Load the explicitly published SFT format without changing tensorization.

Published artifact integrity is checked here. Source approval identities remain
lineage identities; they are never presented as hashes of the redacted files.
All other dataset formats retain the original admission implementation.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from . import dataset_loader

FORMAT = "optiatlas-sanitized-gold-sft-v1"
_native_loader = None
_native_identity = None


def _published(dataset_dir, *, reserved_paths=None, allow_demo=False, validation_workers=1):
    from optics_agent_sft.data import Admission
    from optics_agent_sft.curation.admission import isolation_key, prompt_hash, reservations
    from optics_agent_sft.training.config import canonical_hash

    directory = Path(dataset_dir).expanduser().resolve()
    root = directory.parent.parent
    if directory != root / "agent_sft/sft":
        raise ValueError("Published SFT must retain the dataset-root/agent_sft/sft layout")
    proof = dataset_loader.validate_sft(root, workers=validation_workers)
    manifest = json.loads((directory / "dataset_manifest.json").read_text())
    embedded = [dataset_loader.local(directory, rel) for rel in manifest.get("reserved_manifests", [])]
    reserved = reservations([*(reserved_paths or []), *embedded])
    samples = []
    grouped = defaultdict(list)
    for split in ("train", "validation", "demo"):
        for row in dataset_loader.load_sft(root, split):
            if row.get("is_demo") and not allow_demo:
                raise ValueError("Demo decisions are not allowed")
            # This export convenience field is not consumed by the native
            # tensorizer. Removing its absolute paths keeps data identity
            # independent of the download location, as in native admission.
            row.pop("images", None)
            samples.append(row)
            grouped[row["trajectory_sha256"]].append(row)
    accepted = []
    split_identities = {split: defaultdict(set) for split in ("train", "validation", "demo")}
    source_by_hash = {s["source_sha256"]: s for s in manifest["sources"]}
    for loaded in dataset_loader.load_gold_trajectories(root):
        lineage = loaded["source_sha256"]
        source = source_by_hash[lineage]
        gold = loaded["trajectory"]
        task = gold["task"]
        selected = grouped[lineage]
        if not selected:
            raise ValueError("Published trajectory has no selected decision")
        split = selected[0]["split"]
        if any(r["split"] != split or r["scene_family"] != task["scene_family"]
               or r["task_id"] != task["task_id"] for r in selected):
            raise ValueError("Published trajectory/sample identity mismatch")
        identity = {"task_ids": {isolation_key(task["task_id"])},
                    "scene_families": {isolation_key(task["scene_family"])},
                    "prompt_sha256": {prompt_hash(task["prompt"])},
                    "image_sha256": {item["sha256"] for item in gold["assets"]}}
        for key, values in identity.items():
            if values & reserved[key]:
                raise ValueError("Published SFT overlaps reserved benchmark identity: " + key)
            for other in split_identities:
                if other != split and values & split_identities[other][key]:
                    raise ValueError("Published SFT has cross-split overlap: " + key)
            split_identities[split][key].update(values)
        record = {
            "trajectory_path": str(dataset_loader.local(directory, source["gold"])),
            "trajectory_sha256": lineage,
            "human_review_sha256": None,
            "machine_review_sha256": canonical_hash(gold["evidence_files"]),
            "selected_step_ids": source["selected_step_ids"],
            "scene_family": task["scene_family"], "split": split,
            "original_split": selected[0]["original_split"], "is_demo": selected[0]["is_demo"],
            "models": gold.get("models", {}),
            "artifact_paths": loaded["artifacts"],
            "published_trajectory_sha256": loaded["release_sha256"],
            "published_manifest_sha256": proof["manifest_sha256"],
            "source_manifest_sha256": proof["source_manifest_sha256"],
            "admission_format": FORMAT,
        }
        review = selected[0].get("review")
        if review:
            if any(row.get("review") != review for row in selected):
                raise ValueError("Selected decisions disagree on source review provenance")
            record.update(review)
        accepted.append(record)
    admission = Admission(samples=samples, accepted=accepted, rejected=[])
    admission.published_release_identity = {
        "format": FORMAT, "manifest_sha256": proof["manifest_sha256"],
        "source_manifest_sha256": proof["source_manifest_sha256"],
        "integrity_verified": True,
        "validation_mode": "published_frozen_decisions_with_source_lineage",
    }
    return admission


def load_dataset(dataset_dir, **kwargs):
    manifest_path = Path(dataset_dir).expanduser() / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") == FORMAT:
            return _published(dataset_dir, **kwargs)
    if _native_loader is None:
        from optics_agent_sft.data import load_dataset as native
        return native(dataset_dir, **kwargs)
    return _native_loader(dataset_dir, **kwargs)


def dataset_identity(admission):
    result = _native_identity(admission)
    if hasattr(admission, "published_release_identity"):
        result = {**result, "published_release": admission.published_release_identity}
    return result


def install_sft_loader():
    """Install only at the training data boundary; ordinary formats delegate."""
    global _native_loader, _native_identity
    from optics_agent_sft.training import provenance
    if provenance.load_dataset is load_dataset:
        return
    _native_loader = provenance.load_dataset
    _native_identity = provenance.dataset_identity
    provenance.load_dataset = load_dataset
    provenance.dataset_identity = dataset_identity
