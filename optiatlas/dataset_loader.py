#!/usr/bin/env python3
"""Path-safe readers and integrity checks for the private OptiAtlas dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
from functools import lru_cache


def _json(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


@lru_cache(maxsize=65536)
def _directory(path):
    return Path(path).resolve()


@lru_cache(maxsize=131072)
def local(root, relative):
    root = _directory(str(root))
    path = root / relative
    if Path(relative).is_absolute():
        raise ValueError("Published references must be relative")
    resolved = _directory(str(path.parent)) / path.name
    if resolved.is_symlink():
        resolved = resolved.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Missing or escaping dataset reference: " + str(relative))
    return resolved


def load_generator(root, variant="dense"):
    if variant not in {"dense", "simple", "agent"}:
        raise ValueError("variant must be dense, simple or agent")
    for row in _rows(local(root, f"generator/{variant}/train.jsonl")):
        row["image"] = str(local(root, row["image"]))
        yield row


def load_sft(root, split="train"):
    if split not in {"train", "validation", "demo"}:
        raise ValueError("Unknown split")
    base = Path(root).resolve() / "agent_sft/sft"
    for row in _rows(local(base, split + ".jsonl")):
        row["images"] = [str(local(base, p)) for p in row.get("images", [])]
        for message in row.get("messages", []):
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        part["image_url"]["url"] = str(local(base, part["image_url"]["url"]))
                    elif part.get("type") == "image":
                        part["image"] = str(local(base, part["image"]))
        yield row


def load_gold_trajectories(root):
    """Read complete accepted trajectories with resolved evidence/image maps.

    Logical paths inside each trajectory retain their original identity. The
    returned artifacts dictionary resolves every such path in this release.
    """
    base = Path(root).resolve() / "agent_sft/sft"
    manifest = _json(local(base, "dataset_manifest.json"))
    for source in manifest["sources"]:
        yield {
            "trajectory": _json(local(base, source["gold"])),
            "source_sha256": source["source_sha256"],
            "release_sha256": source["sha256"],
            "selected_step_ids": source["selected_step_ids"],
            "artifacts": {original: str(local(base, relative))
                          for original, relative in source["artifact_map"].items()},
        }


def load_rl(root, split="train"):
    if split not in {"train", "validation"}:
        raise ValueError("Unknown split")
    yield from _rows(local(root, f"agent_rl/{split}.jsonl"))


def load_rl_training(root, split="train"):
    """Return the original training schema, including system messages/tools."""
    import pyarrow.parquet as pq
    if split not in {"train", "validation"}:
        raise ValueError("Unknown split")
    return pq.read_table(local(root, f"agent_rl/{split}.parquet"))


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_sft(root, *, workers=8):
    """Verify the published SFT subpackage, including every image byte.

    This validates a frozen released decision set. It does not rerun the old
    raw-provider admission process, whose source identities remain provenance.
    """
    local.cache_clear(); _directory.cache_clear()
    root = Path(root).resolve()
    base = root / "agent_sft/sft"
    manifest_path = local(base, "dataset_manifest.json")
    manifest = _json(manifest_path)
    if (manifest.get("schema_version") != 1
            or manifest.get("format") != "optiatlas-sanitized-gold-sft-v1"
            or manifest.get("projection_version") != "native-tool-text-image-user-v1"
            or manifest.get("portable") is not True):
        raise ValueError("Unsupported released SFT schema")
    published = _json(local(root, "release_manifest.json"))
    expected_manifest = published["files"]["agent_sft/sft/dataset_manifest.json"]["sha256"]
    manifest_sha = _digest(manifest_path)
    if manifest_sha != expected_manifest:
        raise ValueError("Released SFT manifest differs from publication inventory")
    inventory = manifest["files"]
    def check(item):
        relative, expected = item
        if _digest(local(base, relative)) != expected:
            raise ValueError("Released SFT artifact hash mismatch: " + relative)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(check, inventory.items()):
            pass
    sources = {}
    for source in manifest["sources"]:
        lineage = source["source_sha256"]
        if lineage in sources or source["sha256"] != inventory.get(source["gold"]):
            raise ValueError("Duplicate or unbound published Gold source")
        sources[lineage] = source
        for relative in source["artifact_map"].values():
            if relative not in inventory:
                raise ValueError("Gold artifact outside verified SFT inventory")
    seen = set(); selected = {}; counts = {}
    for split in ("train", "validation", "demo"):
        count = 0
        for row in _rows(local(base, split + ".jsonl")):
            lineage = row["trajectory_sha256"]
            step = row["step_id"]
            if (lineage not in sources or step not in sources[lineage]["selected_step_ids"]
                    or row["sample_id"] in seen or row["split"] != split
                    or row["sample_id"] != "gold:" + lineage + ":" + step):
                raise ValueError("Released SFT row has inconsistent source/selection identity")
            seen.add(row["sample_id"]);selected.setdefault(lineage, []).append(step)
            message_images = [part["image_url"]["url"] for message in row["messages"]
                              if isinstance(message.get("content"), list)
                              for part in message["content"] if part.get("type") == "image_url"]
            if message_images != row.get("images", []) or len(message_images) != len(row["image_sha256"]):
                raise ValueError("Released SFT image order/hash count mismatch")
            if any(inventory.get(path) != digest for path, digest in zip(message_images, row["image_sha256"])):
                raise ValueError("Released SFT image hashes differ from verified artifacts")
            count += 1
        if count != manifest["counts"][split]:
            raise ValueError("Released SFT split count mismatch")
        counts[split] = count
    for lineage, source in sources.items():
        if sorted(selected.get(lineage, [])) != sorted(source["selected_step_ids"]):
            raise ValueError("Published selected Gold steps are incomplete")
    return {"status": "passed", "format": manifest["format"], "manifest_sha256": manifest_sha,
            "source_manifest_sha256": manifest["source_manifest_sha256"], "files_verified": len(inventory),
            "trajectories": len(sources), "counts": counts, "image_bytes_verified": True}


def validate(root, *, hash_images=False, workers=8):
    local.cache_clear()
    _directory.cache_clear()
    root = Path(root).resolve()
    manifest = _json(root / "release_manifest.json")
    files = manifest["files"]
    def check(item):
        relative, entry = item
        path = local(root, relative)
        if path.stat().st_size != entry["bytes"]:
            raise ValueError("File size changed: " + relative)
        if entry["category"] != "image" or hash_images:
            if _digest(path) != entry["sha256"]:
                raise ValueError("File hash changed: " + relative)
            return "image" if entry["category"] == "image" else "metadata"
        return "image_hash_preverified" if entry.get("sha256_status") == "verified" else "image_hash_pending"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verified = Counter(pool.map(check, files.items()))
    counts = {}
    generator_images = set()
    for variant, expected in (("dense", 13867), ("simple", 13867), ("agent", 13866)):
        n = 0
        for row in load_generator(root, variant):
            generator_images.add(row["image"]);n += 1
        if n != expected:
            raise ValueError("Generator count mismatch")
        counts["generator_" + variant] = n
    if len(generator_images) != 13867:
        raise ValueError("Generator images are not shared as expected")
    for split, expected in (("train", 6586), ("validation", 109), ("demo", 0)):
        n = sum(1 for _ in load_sft(root, split))
        if n != expected:
            raise ValueError("SFT count mismatch")
        counts["sft_" + split] = n
    for split, expected in (("train", 3800), ("validation", 200)):
        n = sum(1 for _ in load_rl(root, split))
        if n != expected:
            raise ValueError("RL count mismatch")
        counts["rl_" + split] = n
    sft = _json(root / "agent_sft/sft/dataset_manifest.json")
    if len(sft["sources"]) != 3063:
        raise ValueError("Gold trajectory count mismatch")
    for source in sft["sources"]:
        for relative in [source["gold"], *source["artifact_map"].values()]:
            local(root / "agent_sft/sft", relative)
    for version in ("v6", "v6_1"):
        samples = _json(root / "benchmark" / version / ("light_omni_" + version + ".json"))
        if len(samples) != 457 or sum(len(x["questions"]) for x in samples) != 1002:
            raise ValueError("Benchmark count mismatch")
    return {"status": "passed", "counts": counts, "generator_unique_images": len(generator_images), "complete_gold_trajectories": len(sft["sources"]), "integrity": dict(verified), "image_hashes_reverified_in_this_check": hash_images, "release_images_preverified": manifest.get("image_content_hashes_reverified", False)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--validate", action="store_true", required=True)
    ap.add_argument("--hash-images", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    print(json.dumps(validate(args.root, hash_images=args.hash_images, workers=args.workers), ensure_ascii=False))


if __name__ == "__main__":
    main()
