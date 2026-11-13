"""Explicit local paths and environment-variable API configuration."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    settings.setdefault("paths", {})["code"] = str(ROOT)
    settings.setdefault("runtime", {}).setdefault("python", sys.executable)
    settings["runtime"].setdefault("generator_python", sys.executable)
    for key, value in settings["paths"].items():
        if key != "code":
            settings["paths"][key] = str((Path(path).resolve().parent / Path(os.path.expandvars(value)).expanduser()).resolve())
    api = settings.setdefault("api", {})
    api.setdefault("judge_base_url", os.environ.get("OPTIATLAS_API_BASE_URL", ""))
    api.setdefault("judge_model", os.environ.get("OPTIATLAS_JUDGE_MODEL", ""))
    api.setdefault("judge_endpoint", api["judge_base_url"].rstrip("/") + "/chat/completions")
    api.setdefault("policy_base_url", "http://127.0.0.1:8000/v1")
    api.setdefault("draw_endpoint", "http://127.0.0.1:18320/v1/images/generations")
    return settings


def resolve(value, settings):
    if isinstance(value, dict):
        return {key: resolve(item, settings) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, settings) for item in value]
    if not isinstance(value, str):
        return value
    def replace(match):
        current = settings
        for part in match.group(1).split("."):
            if not isinstance(current, dict) or part not in current:
                raise ValueError("Missing explicit configuration: " + match.group(1))
            current = current[part]
        return str(current)
    return re.sub(r"\$\{((?:paths|runtime|api)\.[^}]+)\}", replace, value)


def configure_imports():
    for path in (ROOT, ROOT / "vendor/verl"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("DRAW_ALIGN_OPSD_ROOT", str(ROOT / "vendor/DiffusionOPSD"))
    os.environ.setdefault("OPTICS_REPO_ROOT", str(ROOT))
