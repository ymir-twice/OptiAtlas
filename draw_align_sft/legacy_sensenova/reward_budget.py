"""Durable pre-call accounting; timeouts count because billing is uncertain."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid


class PersistentRewardBudget:
    def __init__(self, reward, path, *, run_identity, max_requests):
        self.reward, self.path = reward, Path(path)
        self.identity, self.maximum = run_identity, int(max_requests)
        if self.maximum < 1:
            raise ValueError("Reward request budget must be positive")
        self.requests = 0
        if self.path.exists():
            prior = json.loads(self.path.read_text())
            if prior["identity"] != self.identity:
                raise ValueError("Reward budget belongs to a different run")
            self.requests = int(prior["requests"])

    def score(self, prompt, image_path, metadata=None):
        if self.requests >= self.maximum:
            raise RuntimeError("Persistent reward request budget exhausted")
        self.requests += 1
        temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex)
        with temporary.open("x") as handle:
            json.dump({"identity": self.identity, "requests": self.requests}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        return self.reward.score(prompt, image_path, metadata)
