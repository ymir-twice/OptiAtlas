"""Re-admit prepared UMM decisions from reviewed immutable trajectories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from optics_agent_sft.curation.admission import admit_trajectories, read_json_snapshot, strict_json_loads
from optics_experiments.agent_data import sense_rows_from_admission

from .data import sha256_file


def validate_trajectory_dataset(dataset, *, policy="prepared"):
    if policy not in {"prepared", "external"}:
        raise ValueError("data_policy must be prepared or explicitly external")
    path = Path(dataset).resolve()
    manifest_path = path.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        if policy == "external":
            return {"policy": "external", "source_admission": "not_attested"}
        raise ValueError("trajectory_sft requires a prepared dataset_manifest.json; custom external rows require data_policy=external")
    manifest, manifest_hash = read_json_snapshot(manifest_path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows = [strict_json_loads(line) for line in raw.decode().splitlines() if line.strip()]
    record = manifest.get("files", {}).get(path.name, {})
    if (manifest.get("schema_version") != 1 or record.get("sha256") != digest
            or Path(record.get("path", "")).resolve() != path
            or manifest.get("image_target_policy") != "reviewed_final_draw_only"):
        raise ValueError("Prepared Sense dataset does not match its manifest and target policy")
    reserved = manifest.get("reserved_manifests", [])
    if {str(Path(p).resolve()): sha256_file(p) for p in reserved} != manifest.get("reserved_manifest_sha256", {}):
        raise ValueError("Reserved-data manifest changed after Sense preparation")
    sources = manifest.get("admission", {}).get("accepted", [])
    if not sources:
        raise ValueError("Prepared Sense data lacks source trajectory admission evidence")
    fresh = admit_trajectories([item["trajectory_path"] for item in sources], reserved_paths=reserved,
                              accepted_review_types=tuple(manifest.get("review_policy", ["human"])))
    if fresh.rejected or fresh.accepted != sources:
        raise ValueError("Source trajectory, observed image bytes, final target or review changed after Sense preparation")
    expected = [row for row in sense_rows_from_admission(fresh) if row["split"] == "train"]
    if rows != expected or record.get("rows") != len(rows):
        raise ValueError("Prepared Sense decisions differ from re-admitted causal source decisions")
    return {"policy": "prepared", "manifest_sha256": manifest_hash,
            "source_admission_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            "decision_count": len(rows), "dataset_sha256": digest}
