"""Local records are authoritative; W&B is an optional mirror on rank zero."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def redact(value: Any, key: str = "") -> Any:
    """Never resolve environment variables or persist credential-bearing URLs."""
    normalized = key.lower()
    if any(part in normalized for part in ("password", "authorization", "secret")):
        return "<redacted>"
    if normalized in {"api_key", "access_key", "token", "ak", "sk"}:
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str) and value.startswith(("https://", "http://")):
        url = urlsplit(value)
        host = url.netloc.rsplit("@", 1)[-1]
        return urlunsplit((url.scheme, host, url.path, "", ""))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def atomic_json(path: str | Path, value: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _ImageScreenProgress:
    """Console presentation only; structured item records remain authoritative."""

    def __init__(self, fields: dict):
        self.started = time.monotonic()
        self.last_printed: float | None = None
        self.width = 0
        self.line_open = False
        self.finished = False
        self.tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self.interval = max(0.25, float(fields.get("progress_interval_seconds", 10)))
        self.counters = {"total": 0, "selected": 0, "resumed": 0, "completed": 0,
                         "accepted": 0, "rejected": 0, "errors": 0, "inflight": 0}
        self.update(fields)

    def update(self, fields: dict, *, item: bool = False):
        if "elapsed_seconds" in fields:
            self.started = time.monotonic() - max(0.0, float(fields["elapsed_seconds"]))
        for key in self.counters:
            if key in fields:
                self.counters[key] = max(0, int(fields[key]))
        if "completed_this_run" in fields and "completed" not in fields:
            self.counters["completed"] = max(0, int(fields["completed_this_run"]))
        if "unfinished" in fields and "inflight" not in fields:
            self.counters["inflight"] = max(0, int(fields["unfinished"]))
        # Backwards compatibility with callers that only report each item's status.
        status_key = {"accepted": "accepted", "rejected": "rejected", "error": "errors"}.get(fields.get("status"))
        if item and status_key and status_key not in fields:
            self.counters[status_key] += 1

    def newline(self):
        if self.line_open:
            print(file=sys.stderr, flush=True)
            self.line_open = False
            self.width = 0

    def render(self, *, force: bool = False, status: str | None = None):
        now = time.monotonic()
        interval = 0.25 if self.tty else self.interval
        if not force and self.last_printed is not None and now - self.last_printed < interval:
            return
        self.last_printed = now
        counters = self.counters
        total, completed = counters["total"], counters["completed"]
        processed = min(total, counters["resumed"] + completed)
        fraction = processed / total if total else 1.0
        bar_length = 20
        filled = min(bar_length, int(fraction * bar_length))
        bar = "=" * filled + "-" * (bar_length - filled)
        elapsed = max(0, now - self.started)
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, counters["selected"] - completed)
        eta = remaining / rate if rate > 0 else 0 if not remaining else None
        line = (f"image_screen [{bar}] {fraction:6.1%} {processed}/{total}"
                f" accepted={counters['accepted']} rejected={counters['rejected']}"
                f" errors={counters['errors']} resumed={counters['resumed']}"
                f" run={completed}/{counters['selected']} inflight={counters['inflight']}"
                f" elapsed={_duration(elapsed)} rate={rate:.2f}/s ETA={_duration(eta)}")
        if status:
            line += f" status={status}"
        if self.tty:
            columns = max(1, shutil.get_terminal_size(fallback=(240, 24)).columns - 1)
            if len(line) > columns:
                # Reserve the last column: implicit wrapping defeats carriage-return
                # redraws. Short labels retain the useful counters on an 80-column TTY.
                short_elapsed = _duration(elapsed).removeprefix("00:")
                short_eta = _duration(eta).removeprefix("00:")
                line = (f"{fraction:.1%} {processed}/{total} A={counters['accepted']}"
                        f" R={counters['rejected']} E={counters['errors']}"
                        f" t={short_elapsed} {rate:.2g}/s ETA={short_eta}")
                if status:
                    line += f" {status}"
                available = columns - len(line) - 3  # Brackets and separating space.
                if available >= 2:
                    bar_length = min(12, available)
                    filled = min(bar_length, int(fraction * bar_length))
                    line = "[" + "=" * filled + "-" * (bar_length - filled) + "] " + line
                line = line[:columns]
            sys.stderr.write("\r" + line + " " * max(0, min(self.width, columns) - len(line)))
            sys.stderr.flush()
            self.width = len(line)
            self.line_open = True
        else:
            print(line, file=sys.stderr, flush=True)


class RunLogger:
    def __init__(self, output_dir, config: dict, *, rank: int | None = None):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else rank
        self.started = time.monotonic()
        self.wandb_run = None
        self.enabled = self.rank == 0
        self.closed = False
        self.console_enabled = config.get("logging", {}).get("console", True) is not False
        self._image_screen_progress: _ImageScreenProgress | None = None
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        versions = {"python": platform.python_version()}
        for name in ("torch", "transformers", "peft", "hydra-core", "Pillow", "wandb"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        safe = redact(config)
        self.run_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(config.get("run_id") or
                            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")))
        self.record_dir = self.output_dir / "records" / self.run_id
        self.record_dir.mkdir(parents=True, exist_ok=False)
        self.manifest = {"run_id": self.run_id, "started_at": utcnow(), "status": "running",
                         "config_sha256": fingerprint(safe), "versions": versions,
                         "world_size": int(os.environ.get("WORLD_SIZE", "1"))}
        atomic_json(self.record_dir / "config.json", safe)
        atomic_json(self.record_dir / "run.json", self.manifest)
        atomic_json(self.output_dir / "latest_run.json", {"run_id": self.run_id,
                    "records": str(self.record_dir)})
        wandb_config = config.get("logging", {}).get("wandb", {})
        if wandb_config.get("enabled", False):
            import wandb
            self.wandb_run = wandb.init(project=wandb_config.get("project", "light-omni"),
                entity=wandb_config.get("entity"), name=wandb_config.get("name") or self.run_id,
                group=wandb_config.get("group"), tags=wandb_config.get("tags", []),
                mode=wandb_config.get("mode", "offline"), config=safe,
                dir=str(self.record_dir), id=wandb_config.get("id"),
                resume=wandb_config.get("resume", "never"))
        self.progress("run_started", task=config.get("task"), records=str(self.record_dir))

    def _append(self, filename, record):
        if self.enabled:
            with (self.record_dir / filename).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(redact(record), ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()

    def log(self, metrics: dict, step: int | None = None):
        if not self.enabled:
            return
        record = {"time": utcnow(), "elapsed_seconds": time.monotonic() - self.started,
                  "step": step, **metrics}
        self._append("metrics.jsonl", record)
        if self.wandb_run is not None:
            self.wandb_run.log(redact(metrics), step=step)

    def event(self, name: str, **fields):
        self._append("events.jsonl", {"time": utcnow(), "event": name, **fields})

    def progress(self, name: str, **fields):
        """Flush concise progress to stderr and disk without changing JSON stdout.

        Callers pass structured counters, paths and typed errors, never raw HTTP
        messages or model responses. The same credential redaction as other run
        records applies to every destination.
        """
        if not self.enabled:
            return
        timestamp = utcnow()
        safe = redact(fields)
        self._append("events.jsonl", {"time": timestamp, "event": name, **safe})
        detail = " ".join(f"{key}={json.dumps(value, ensure_ascii=False, allow_nan=False)}"
                          for key, value in safe.items())
        line = f"{timestamp} {name}" + (f" {detail}" if detail else "")
        with (self.record_dir / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        if self.console_enabled:
            self._print_progress(name, safe, line)

    def _print_progress(self, name: str, fields: dict, line: str):
        if name == "image_screen_started":
            if self._image_screen_progress:
                self._image_screen_progress.newline()
            self._image_screen_progress = _ImageScreenProgress(fields)
            self._image_screen_progress.render(force=True)
            return
        screen = self._image_screen_progress
        screen_events = {"image_screen_item_completed", "image_screen_waiting", "image_screen_finished",
                         "image_screen_interrupted", "image_screen_failed"}
        if screen and name in screen_events:
            screen.update(fields, item=name == "image_screen_item_completed")
            finished = name in {"image_screen_finished", "image_screen_interrupted", "image_screen_failed"}
            status = fields.get("status") if finished else None
            if finished and status is None:
                status = {"image_screen_interrupted": "interrupted", "image_screen_failed": "failed"}.get(name)
                if status is None:
                    status = "completed" if fields.get("complete") else "incomplete"
            if finished:
                screen.counters["inflight"] = 0
            screen.render(force=finished, status=status)
            if finished:
                screen.newline()
                screen.finished = True
            return
        if screen:
            # A warning or unrelated progress event must not overwrite the bar.
            screen.newline()
        print(line, file=sys.stderr, flush=True)

    def close(self, status="completed"):
        if not self.enabled or self.closed:
            return
        self.closed = True
        if self.console_enabled and self._image_screen_progress:
            screen = self._image_screen_progress
            if not screen.finished:
                screen.render(force=True, status=status)
                screen.finished = True
            screen.newline()
        self.manifest.update(status=status, finished_at=utcnow(),
                             elapsed_seconds=time.monotonic() - self.started)
        atomic_json(self.record_dir / "run.json", self.manifest)
        self.progress("run_finished", status=status,
                      elapsed_seconds=round(self.manifest["elapsed_seconds"], 2))
        if self.wandb_run is not None:
            self.wandb_run.finish(exit_code=0 if status == "completed" else 1)
