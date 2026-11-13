"""Explicit target records; observed images can never silently become targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps
import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("message role must be system/user/assistant/tool")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("message content must be text or typed parts")
        for part in content:
            kind = part.get("type")
            if kind == "text" and isinstance(part.get("text"), str):
                continue
            if kind == "image" and isinstance(part.get("path"), str):
                continue
            raise ValueError("content parts must be {type:text,text} or {type:image,path}")


def normalize_record(row, root: Path):
    row = dict(row)
    if not isinstance(row.get("id"), (str, int)):
        raise ValueError("Each training record requires a stable id")
    kind = row.get("kind", "image" if "caption" in row else None)
    if kind not in {"image", "text"}:
        raise ValueError("Each trajectory decision must declare kind=image or kind=text")
    row["kind"] = kind
    if "messages" not in row:
        caption = row.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Atomic image records require a nonempty caption")
        row["messages"] = [{"role": "user", "content": caption}]
    validate_messages(row["messages"])
    row["messages"] = json.loads(json.dumps(row["messages"]))
    for message in row["messages"]:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part["type"] == "image":
                    part["path"] = str((root / part["path"]).resolve())
                    if not Path(part["path"]).is_file():
                        raise FileNotFoundError(part["path"])
    if kind == "image":
        target = row.get("target_image", row.get("image"))
        if not isinstance(target, str):
            raise ValueError("Image supervision requires an explicit target_image or image")
        row["target_image"] = str((root / target).resolve())
        if not Path(row["target_image"]).is_file():
            raise FileNotFoundError(row["target_image"])
        observed = [p["path"] for m in row["messages"] if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image"]
        if row["target_image"] in observed:
            raise ValueError("The supervised image target is already present in its causal context")
    elif not isinstance(row.get("target_text"), str) or not row["target_text"].strip():
        raise ValueError("Text supervision requires nonempty target_text")
    return row


def load_records(dataset, *, require_split="train"):
    path = Path(dataset).resolve()
    rows, seen, images = [], set(), set()
    with path.open(encoding="utf8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if require_split and raw.get("split", "train") != require_split:
                continue
            row = normalize_record(raw, path.parent)
            identity = str(row["id"])
            if identity in seen:
                raise ValueError(f"Duplicate decision id at line {line_no}: {identity}")
            seen.add(identity)
            if row["kind"] == "image":
                images.add(row["target_image"])
            for message in row["messages"]:
                if isinstance(message["content"], list):
                    images.update(p["path"] for p in message["content"] if p["type"] == "image")
            rows.append(row)
    if not rows:
        raise ValueError("No eligible training records")
    fingerprint = hashlib.sha256()
    fingerprint.update(sha256_file(path).encode())
    for image in sorted(images):
        fingerprint.update(image.encode())
        fingerprint.update(sha256_file(image).encode())
    return rows, fingerprint.hexdigest()


def image_tensor(path, *, max_pixels: int, multiple: int, device=None, dtype=torch.float32):
    """Preserve the whole optical scene; resize without random crop or reflection."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        scale = min(1.0, (max_pixels / (image.width * image.height)) ** 0.5)
        width = max(multiple, int(image.width * scale) // multiple * multiple)
        height = max(multiple, int(image.height * scale) // multiple * multiple)
        if width * height > max_pixels:
            raise ValueError("max_pixels is smaller than one image token")
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def save_image_tensor(pixels, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = ((pixels.detach().float().cpu()[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
    Image.fromarray(array).save(path)
    return path
