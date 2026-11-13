"""Keep complete multiline captions and identify examples independently of tokenizers."""
from __future__ import annotations

import hashlib
import json
import math
import stat
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_file_stat(path):
    """Metadata used to reject changes after an image's content was checked."""
    info = Path(path).stat()
    if not stat.S_ISREG(info.st_mode):
        raise FileNotFoundError(path)
    return {name: getattr(info, "st_" + name)
            for name in ("dev", "ino", "size", "mtime_ns", "ctime_ns")}


def _checked_image_digest(path, require_images, hash_images):
    if path is None:
        return None, None
    try:
        before = image_file_stat(path)
    except (FileNotFoundError, NotADirectoryError):
        if require_images:
            raise FileNotFoundError(path) from None
        return None, None
    if not hash_images:
        return None, None
    digest = file_digest(path)
    try:
        after = image_file_stat(path)
    except (FileNotFoundError, NotADirectoryError):
        raise ValueError(f"Image changed while computing its SHA256: {path}") from None
    if before != after:
        raise ValueError(f"Image changed while computing its SHA256: {path}")
    return digest, after


class ManifestPathResolver:
    """Canonicalize image paths while resolving each shared parent only once.

    Create one instance per manifest read: directory aliases may change between
    reads. The final filename is still checked for symlinks on every call, and
    their targets (or a final ``..`` component) use pathlib's full resolution.
    Image contents and file existence are never cached.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self._parents = {}

    def __call__(self, image):
        path = self.directory / image
        parent = path.parent
        if parent not in self._parents:
            self._parents[parent] = parent.resolve()
        resolved = self._parents[parent] / path.name
        if path.name == ".." or resolved.is_symlink():
            return resolved.resolve()
        return resolved


def load_pairs(path, *, require_images=True, split="train", hash_images=True,
               progress=None, hash_workers=8):
    """Read one manifest snapshot and hash images with bounded parallel I/O.

    Results and the fingerprint always follow manifest order. ``progress`` is
    called on this thread with keyword fields; the initial ``records_total`` is
    ``None`` until manifest parsing finishes. Set ``hash_workers=1`` for serial
    I/O on storage where concurrency is undesirable.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training JSONL does not exist: {path}")
    if not isinstance(hash_workers, int) or isinstance(hash_workers, bool) or hash_workers < 1:
        raise ValueError("hash_workers must be a positive integer")
    last_progress = time.monotonic()

    def report(completed, total, *, phase="images", force=False):
        nonlocal last_progress
        now = time.monotonic()
        if progress is not None and (force or now - last_progress >= 10):
            progress(path=str(path), records_completed=completed,
                     records_total=total, hash_images=hash_images, phase=phase)
            last_progress = now

    report(0, None, phase="manifest", force=True)
    rows, seen = [], set()
    manifest_bytes = path.read_bytes()
    fingerprint = hashlib.sha256(manifest_bytes)
    resolve_image = ManifestPathResolver(path.parent)
    for line_number, line in enumerate(manifest_bytes.decode("utf8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number}: expected a JSON object")
        if row.get("split", "train") != split:
            continue
        caption = row.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Line {line_number}: a complete nonempty caption string is required")
        identity = str(row.get("id", row.get("sha256", "")))
        if not identity or identity in seen:
            raise ValueError(f"Line {line_number}: missing or duplicate id")
        seen.add(identity)
        image = row.get("image", row.get("target_image"))
        if require_images and not isinstance(image, str):
            raise ValueError(f"Line {line_number}: image path is required for FM supervision")
        resolved = resolve_image(image) if image else None
        if require_images and resolved is None:
            raise ValueError(f"Line {line_number}: image path is required for FM supervision")
        rows.append({**row, "id": identity, "caption": caption,
                     "image": str(resolved) if resolved else None,
                     "_image_sha256": None, "_image_stat": None,
                     "prompt_id": hashlib.sha256(caption.encode("utf8")).hexdigest()})
        report(len(rows), None, phase="manifest")
    if not rows:
        raise ValueError(f"No {split} records in {path}. Export selected caption–image pairs first; "
                         "caption_drafts.jsonl is not automatically a final training manifest.")
    report(0, len(rows), force=True)

    def check(row):
        return _checked_image_digest(row["image"], require_images, hash_images)

    def retain(index, result):
        digest, image_stat = result
        rows[index]["_image_sha256"] = digest
        rows[index]["_image_stat"] = image_stat
        if digest is not None:
            fingerprint.update(digest.encode())
        report(index + 1, len(rows), force=index + 1 == len(rows))

    if hash_workers == 1 or not hash_images:
        for index, row in enumerate(rows):
            retain(index, check(row))
    else:
        # Keep at most two tasks per worker queued, rather than eagerly creating
        # tens of thousands of Future objects on large manifests.
        workers = min(hash_workers, len(rows))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="image-sha256") as executor:
            pending = deque()
            next_row = 0
            for _ in range(min(len(rows), workers * 2)):
                pending.append(executor.submit(check, rows[next_row]))
                next_row += 1
            index = 0
            while pending:
                retain(index, pending.popleft().result())
                index += 1
                if next_row < len(rows):
                    pending.append(executor.submit(check, rows[next_row]))
                    next_row += 1
    return rows, fingerprint.hexdigest()


def image_dimensions(path, max_pixels, multiple):
    """Whole scene, no crops/flips; axes round to the model's native token multiple."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
    if min(width, height, max_pixels, multiple) <= 0:
        raise ValueError("Image dimensions, max_pixels and multiple must be positive")
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    width = max(multiple, int(width * scale) // multiple * multiple)
    height = max(multiple, int(height * scale) // multiple * multiple)
    if width * height > max_pixels:
        raise ValueError("max_pixels cannot accommodate this image's native token grid")
    return height, width


def scheduled_rows(rows, step, count, seed, rank=0, world_size=1):
    import random
    selected = []
    for offset in range(count):
        index = step * count * world_size + offset * world_size + rank
        epoch, position = divmod(index, len(rows))
        order = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(order)
        selected.append(rows[order[position]])
    return selected


def scheduled_epoch_rows(rows, step, count, seed, rank=0, world_size=1):
    """Deterministic complete epochs, padding only the final global batch.

    Padding repeats the beginning of the same shuffled epoch. No sample from
    the next epoch is consumed early, so changing batch size cannot skip data.
    Resume reconstructs the same permutation from the optimizer step alone.
    """
    if not rows or min(count, world_size) < 1 or step < 0 or not 0 <= rank < world_size:
        raise ValueError("Epoch scheduling requires data, positive counts and a valid step/rank")
    import random
    global_batch = count * world_size
    steps_per_epoch = (len(rows) + global_batch - 1) // global_batch
    epoch, within_epoch = divmod(step, steps_per_epoch)
    order = list(range(len(rows)))
    random.Random(seed + epoch).shuffle(order)
    start = within_epoch * global_batch
    return [rows[order[(start + offset * world_size + rank) % len(rows)]] for offset in range(count)]
