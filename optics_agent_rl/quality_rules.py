"""Small observable quality rubrics derived from the training taxonomy only."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

RULES_PATH = Path(__file__).with_name("quality_rules_37.json")
_GENERIC_RULES = [
    {"rule_id": "R1", "text": "The requested optical phenomenon has visible evidence in the submitted image."},
    {"rule_id": "R2", "text": "Visible illumination, materials, interfaces and receiver/viewpoint relationships are mutually consistent."},
    {"rule_id": "R3", "text": "The phenomenon is not replaced by an unrelated lookalike, decorative overlay or impossible visible relation."},
]


@lru_cache(maxsize=1)
def _catalog():
    value = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    if len(value["categories"]) != 37:
        raise ValueError("Quality rubric must cover the 37 training categories")
    return value


def quality_rubric(category_id=None):
    """Unknown explicit categories fail closed; None supports service preflight."""
    catalog = _catalog()
    if category_id is None:
        entry = {"name": "General optical scene (service preflight or legacy metadata)", "rules": _GENERIC_RULES}
    elif not isinstance(category_id, str) or category_id not in catalog["categories"]:
        raise ValueError("Unknown optical quality category")
    else:
        entry = catalog["categories"][category_id]
    result = {"version": catalog["version"], "category_id": category_id, **entry}
    # Never expose the mutable cached catalog to callers.
    return json.loads(json.dumps(result, ensure_ascii=False))


def rubric_digest(rubric):
    return hashlib.sha256(json.dumps(rubric, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()
