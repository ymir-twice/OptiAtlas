"""Portable curated JSONL, actual assets, and separately retained raw provenance."""
from __future__ import annotations
import copy
import json
import shutil
from pathlib import Path

from .admission import (MAX_JSON_BYTES, Admission, admit_trajectories, local_artifact,
                        read_json, sha256_file, strict_json_loads)
from .io_helpers import validate_workers


def _portable_row(sample, path_map):
    row = copy.deepcopy(sample)
    images = []
    for message in row["messages"]:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    original = part["image_url"]["url"]
                    if original not in path_map:
                        raise ValueError("Missing exported image mapping")
                    part["image_url"]["url"] = path_map[original]
                    images.append(path_map[original])
    row["images"] = images
    return row


def export_dataset(admission: Admission, output_dir: str | Path, *, reserved_paths=None,
                   copy_file_fn=None) -> dict:
    if not admission.samples:
        raise ValueError("No admitted native decision samples")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Dataset output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    files, sources, path_map = {}, [], {}
    def copy_file(source, relative):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            (copy_file_fn or shutil.copyfile)(source, target)
        digest = sha256_file(target)
        if digest != sha256_file(Path(source)):
            raise ValueError("File changed while preparing the dataset")
        files[relative] = digest
    for accepted in admission.accepted:
        trajectory_path = Path(accepted["trajectory_path"])
        digest = accepted["trajectory_sha256"]
        trajectory_relative = f"provenance/{digest}/trajectory.json"
        copy_file(trajectory_path, trajectory_relative)
        if files[trajectory_relative] != digest:
            raise ValueError("Trajectory changed after human admission")
        artifact_specs = {item["path"]: item for item in
                          read_json(output / trajectory_relative).get("artifacts", [])}
        source = {"trajectory": trajectory_relative, "trajectory_sha256": digest,
                  "is_demo": accepted["is_demo"], "artifact_map": {}}
        if accepted["human_review_sha256"]:
            review_relative = f"provenance/{digest}/human_review.json"
            copy_file(trajectory_path.with_name("human_review.json"), review_relative)
            if files[review_relative] != accepted["human_review_sha256"]:
                raise ValueError("Human review changed during export")
            source["human_review"] = review_relative
            source["review_type"] = "human"
        if accepted.get("machine_review_sha256"):
            review_relative = f"provenance/{digest}/machine_review.json"
            copy_file(trajectory_path.with_name("machine_review.json"), review_relative)
            if files[review_relative] != accepted["machine_review_sha256"]:
                raise ValueError("Machine review changed during export")
            source["machine_review"] = review_relative
            source["review_type"] = "model"
        for original, actual in accepted["artifact_paths"].items():
            sha = accepted["artifact_sha256"][actual]
            suffix = Path(original).suffix.lower()
            is_record = artifact_specs.get(original, {}).get("kind") == "research_result"
            if is_record:
                if suffix != ".json":
                    raise ValueError("Research result export requires a JSON record")
                relative = f"provenance/{digest}/records/{sha}.json"
            else:
                if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                    raise ValueError("Unsupported image suffix in export")
                relative = f"assets/{sha}{suffix}"
            copy_file(Path(actual), relative)
            if files[relative] != sha:
                raise ValueError("Artifact changed after admission")
            if not is_record:
                path_map[actual] = relative
            source["artifact_map"][original] = relative
        sources.append(source)
    for split in ("train", "validation", "demo"):
        relative = f"{split}.jsonl"
        with (output / relative).open("w", encoding="utf-8") as handle:
            for sample in admission.samples:
                if sample["split"] == split:
                    handle.write(json.dumps(_portable_row(sample, path_map), ensure_ascii=False, allow_nan=False) + "\n")
        files[relative] = sha256_file(output / relative)
    embedded_reservations = []
    for path in reserved_paths or []:
        relative = f"reservations/{sha256_file(Path(path))}.json"
        copy_file(Path(path), relative)
        if relative not in embedded_reservations:
            embedded_reservations.append(relative)
    review_policy = sorted({item["review_type"] for item in sources if item.get("review_type")}) or ["human"]
    manifest = {"schema_version": 2, "format": "optics-native-agent-sft", "portable": True,
                "review_policy": review_policy, "reserved_manifests": embedded_reservations,
                "projection_version": "native-tool-text-image-user-v1", "sources": sources,
                "files": files, "counts": {split: sum(s["split"] == split for s in admission.samples)
                                             for split in ("train", "validation", "demo")},
                "loss_scope": "one selected native assistant message, including tool_calls and turn end",
                "source_scope": "Raw trajectories/reviews are provenance only, never appended to model inputs",
                "rejected": admission.rejected}
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_dataset(dataset_dir: str | Path, *, allow_demo=False, reserved_paths=None,
                 validation_workers=1) -> Admission:
    validate_workers(validation_workers)
    root = Path(dataset_dir).expanduser().resolve()
    manifest = read_json(root / "dataset_manifest.json")
    if manifest.get("format") == "optics-gold-agent-sft":
        from .gold import load_gold_dataset
        return load_gold_dataset(root, reserved_paths=reserved_paths, validation_workers=validation_workers)
    if manifest.get("schema_version") != 2 or manifest.get("format") != "optics-native-agent-sft" or manifest.get("portable") is not True:
        raise ValueError("Legacy/nonportable datasets are rejected; prepare schema 2 native runs again")
    if manifest.get("projection_version") != "native-tool-text-image-user-v1":
        raise ValueError("Dataset projection version mismatch")
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Dataset manifest needs a complete file hash inventory")
    for relative, digest in manifest.get("files", {}).items():
        if Path(relative).is_absolute() or sha256_file(local_artifact(root, relative)) != digest:
            raise ValueError("Dataset file SHA256 mismatch")
    review_policy = manifest.get("review_policy", ["human"])
    if (not isinstance(review_policy, list) or not review_policy
            or any(not isinstance(value, str) or value not in {"human", "model"} for value in review_policy)
            or len(review_policy) != len(set(review_policy))):
        raise ValueError("Dataset review_policy must explicitly select human and/or model")
    embedded_reservations = manifest.get("reserved_manifests", [])
    if not isinstance(embedded_reservations, list):
        raise ValueError("Dataset reserved_manifests must be a list")
    reservation_paths = list(reserved_paths or [])
    for relative in embedded_reservations:
        if not isinstance(relative, str) or relative not in manifest["files"]:
            raise ValueError("Reserved manifest missing from hash inventory")
        reservation_paths.append(local_artifact(root, relative))
    paths, maps, path_map = [], {}, {}
    for source in manifest.get("sources", []):
        if source["trajectory"] not in manifest["files"]:
            raise ValueError("Source trajectory missing from hash inventory")
        path = local_artifact(root, source["trajectory"])
        if sha256_file(path) != source["trajectory_sha256"]:
            raise ValueError("Source trajectory hash mismatch")
        paths.append(path)
        if not source.get("is_demo"):
            review_kind = source.get("review_type", "human")
            if review_kind not in review_policy:
                raise ValueError("Source review_type is outside dataset review_policy")
            field = "human_review" if review_kind == "human" else "machine_review"
            review = str(Path(source["trajectory"]).with_name(field + ".json"))
            if source.get(field) != review or review not in manifest["files"]:
                raise ValueError(f"{field} missing from hash inventory")
        maps[str(path)] = {}
        for original, relative in source["artifact_map"].items():
            if relative not in manifest["files"]:
                raise ValueError("Artifact missing from hash inventory")
            actual = local_artifact(root, relative)
            maps[str(path)][original] = actual
            path_map[str(actual)] = relative
    admitted = admit_trajectories(paths, allow_demo=allow_demo, reserved_paths=reservation_paths,
                                 artifact_maps=maps, accepted_review_types=tuple(review_policy))
    # Re-derive every selected action from the raw SHA-bound human approval. A
    # modified curated row cannot become a training target merely by updating its hash.
    expected = {row["sample_id"]: _portable_row(row, path_map) for row in admitted.samples}
    actual_rows, seen = [], set()
    for split in ("train", "validation", "demo"):
        if f"{split}.jsonl" not in manifest["files"]:
            raise ValueError("Curated JSONL missing from hash inventory")
        path = local_artifact(root, f"{split}.jsonl")
        count = 0
        with path.open("rb") as stream:
            while line := stream.readline(MAX_JSON_BYTES + 1):
                if len(line) > MAX_JSON_BYTES:
                    raise ValueError("Curated JSONL row exceeds the 64 MiB admission limit")
                count += 1
                row = strict_json_loads(line.decode("utf-8"))
                if row.get("schema_version") != 2 or row.get("split") != split:
                    raise ValueError("Curated sample schema/split mismatch")
                if row.get("is_demo") and split != "demo":
                    raise ValueError("Demo appeared in a training/validation file")
                if row["sample_id"] in seen:
                    raise ValueError("Duplicate curated sample ID")
                seen.add(row["sample_id"])
                if row.get("is_demo") and not allow_demo:
                    continue
                if expected.get(row["sample_id"]) != row:
                    raise ValueError("Curated sample no longer matches approved raw trajectory")
                actual_rows.append(row)
        if manifest.get("counts", {}).get(split) != count:
            raise ValueError("Dataset count does not match curated JSONL")
    if {row["sample_id"] for row in actual_rows} != set(expected):
        raise ValueError("Curated samples missing or extra relative to human selection")
    # Re-admission sorts content-addressed provenance paths. Preserve the
    # validated JSONL order while retaining re-derived absolute image paths.
    samples_by_id = {row["sample_id"]: row for row in admitted.samples}
    admitted.samples = [samples_by_id[row["sample_id"]] for row in actual_rows]
    source_order = {str(path): index for index, path in enumerate(paths)}
    admitted.accepted.sort(key=lambda source: source_order[source["trajectory_path"]])
    return admitted
