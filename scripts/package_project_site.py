#!/usr/bin/env python3
"""Check and optionally ZIP the standalone site, with no project dependencies."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SUFFIXES = {".html", ".css", ".js", ".json", ".png", ".jpg", ".jpeg", ".webp",
            ".svg", ".ico", ".mp4", ".webm", ".vtt", ".woff", ".woff2", ".txt"}


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.references: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        for name in ("href", "src", "poster"):
            if values.get(name):
                self.references.append(values[name])


def local_target(source: Path, reference: str) -> Path | None:
    link = urlsplit(reference)
    if link.scheme or link.netloc or not link.path:
        return None
    if link.path.startswith("/"):
        raise ValueError(f"Use a relative asset path for project Pages: {reference}")
    target = (source.parent / unquote(link.path)).resolve()
    if not target.is_relative_to(SITE.resolve()):
        raise ValueError(f"Page points outside site/: {source.name}: {reference}")
    return target


def validate() -> list[Path]:
    if SITE.is_symlink() or not (SITE / "index.html").is_file():
        raise ValueError("site/ must be a real directory containing index.html")
    files = []
    for path in sorted(SITE.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks cannot be published: {path.relative_to(SITE)}")
        if not path.is_file():
            continue
        relative = path.relative_to(SITE)
        if path.name != ".nojekyll" and (any(p.startswith(".") for p in relative.parts)
                                           or path.suffix.lower() not in SUFFIXES):
            raise ValueError(f"Unexpected website file: {relative}")
        files.append(path)
    for path in files:
        refs: list[str] = []
        if path.suffix == ".html":
            parser = AssetParser()
            parser.feed(path.read_text(encoding="utf-8"))
            refs = parser.references
            for ref in refs:
                if ref.startswith("#") and len(ref) > 1 and unquote(ref[1:]) not in parser.ids:
                    raise ValueError(f"Missing section in {path.name}: {ref}")
        elif path.suffix == ".css":
            refs = [m.strip().strip("\"'") for m in re.findall(r"url\(([^)]+)\)", path.read_text())]
        elif path.suffix == ".json":
            json.loads(path.read_text())
        for ref in refs:
            target = local_target(path, ref)
            if target is not None and not target.is_file():
                raise ValueError(f"Missing local resource in {path.name}: {ref}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without publishing")
    parser.add_argument("--output", type=Path, help="Write a standalone ZIP to this path")
    args = parser.parse_args()
    files = validate()
    if args.output:
        output = args.output.expanduser().absolute()
        if output.resolve().is_relative_to(SITE.resolve()):
            raise ValueError("ZIP output must be outside site/")
        if output.exists():
            raise ValueError("Output already exists; choose a new path")
        output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, path.relative_to(SITE).as_posix())
    print(json.dumps({"status": "validated", "files": len(files),
                      "bytes": sum(path.stat().st_size for path in files),
                      "zip": str(args.output) if args.output else None}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Website preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
