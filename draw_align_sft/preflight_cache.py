"""One-launch handoff of checked FM data, never a persistent unchecked cache."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

from .data import file_digest

PATH_ENV = "DRAW_ALIGN_PREFLIGHT_SNAPSHOT"
SHA_ENV = "DRAW_ALIGN_PREFLIGHT_SHA256"


def cacheable(config):
    return (config.get("algorithm", {}).get("name") == "flow_matching"
            and not config["algorithm"].get("backend")
            and config.get("data", {}).get("hash_images", True))


def config_digest(config):
    # Launcher UUID/path is transport metadata, not part of checkpoint identity.
    stable = {key: value for key, value in config.items() if key not in {"task", "launcher", "hydra"}}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def manifest_hashes(config):
    paths = [config["data"]["path"]]
    if config.get("evaluation", {}).get("enabled", False):
        paths.append(config["evaluation"]["path"])
    return {str(Path(path).expanduser().resolve()): file_digest(path) for path in paths}


def write_snapshot(config, info, rows, directory, *, sources):
    if manifest_hashes(config) != sources:
        raise ValueError("Training/validation JSONL changed during launcher preflight; restart with stable data")
    if any(not row.get("_image_sha256") or not row.get("_image_stat") for row in rows):
        raise ValueError("Preflight handoff requires actual image hashes and file metadata")
    document = {"version": 1, "config_sha256": config_digest(config), "manifests": sources,
                "info": info, "rows": rows}
    content = (json.dumps(document, ensure_ascii=False) + "\n").encode("utf8")
    path = Path(directory) / "preflight.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return {PATH_ENV: str(path), SHA_ENV: hashlib.sha256(content).hexdigest()}


def load_snapshot(config, *, progress=None):
    """Verify manifests and image metadata; do not read all image bytes again.

    Only the launcher's children inherit this reference. Images are still read
    for training, and held-out contents are checked again by FixedFMEvaluator.
    A metadata mismatch fails closed and requires a fresh complete preflight.
    """
    path, expected = os.environ.get(PATH_ENV), os.environ.get(SHA_ENV)
    if path is None and expected is None:
        return None
    if not path or not expected:
        raise ValueError("Incomplete launcher preflight handoff")
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Launcher preflight snapshot integrity check failed")
    document = json.loads(content)
    if document.get("version") != 1 or document.get("config_sha256") != config_digest(config):
        raise ValueError("Launcher preflight configuration differs from worker configuration")
    if manifest_hashes(config) != document["manifests"]:
        raise ValueError("Training/validation JSONL changed after launcher preflight")
    rows, info = document["rows"], document["info"]
    if len(rows) != info["records"]:
        raise ValueError("Launcher preflight record count mismatch")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if info.get("training_plan", {}).get("world_size") != world:
        raise ValueError("Launcher preflight world size differs from worker world size")
    from .data import image_file_stat
    images = {row["image"]: row["_image_stat"] for row in rows}
    def verify(item):
        image, expected_stat = item
        if image_file_stat(image) != expected_stat:
            raise ValueError(f"Image changed after launcher preflight: {image}; restart to recheck image contents")
    total, completed, last = len(images), 0, time.monotonic()
    if progress:
        progress(records_completed=0, records_total=total, snapshot=path)
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="preflight-stat") as pool:
        for _ in pool.map(verify, images.items()):
            completed += 1
            if progress and (time.monotonic() - last >= 10 or completed == total):
                progress(records_completed=completed, records_total=total, snapshot=path)
                last = time.monotonic()
    return info, rows
