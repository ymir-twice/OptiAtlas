"""Rank-zero JSONL is always written; W&B is an optional mirror."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class Logger:
    def __init__(self, output, config, rank=0):
        self.rank = rank
        self.run = None
        self.stream = None
        if rank != 0:
            return
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.stream = (output / "metrics.jsonl").open("a", encoding="utf8", buffering=1)
        self.events = (output / "events.jsonl").open("a", encoding="utf8", buffering=1)
        (output / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        options = config.get("logging", {}).get("wandb", {})
        if options.get("enabled"):
            import wandb
            self.run = wandb.init(project=options.get("project", "Draw-Align-SFT"),
                                  name=options.get("name"), mode=options.get("mode", "offline"),
                                  dir=str(output), config=config)

    def event(self, event, **fields):
        if self.rank:
            return
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        self.events.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(record["time"], event, " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)

    def log(self, values, step, total):
        if self.rank:
            return
        self.stream.write(json.dumps({"step": step, **values}, allow_nan=False) + "\n")
        if self.run is not None:
            self.run.log(values, step=step)
        fraction = min(1.0, step / total)
        filled = int(20 * fraction)
        print(f"train [{'=' * filled}{'-' * (20-filled)}] {step}/{total} "
              + " ".join(f"{key}={values[key]:.6g}" for key in
                         ("loss/total", "reward/mean", "optimizer/lr", "timing/update_seconds") if key in values),
              flush=True)

    def close(self):
        if self.run is not None:
            self.run.finish()
        if self.stream is not None:
            self.stream.close()
            self.events.close()
