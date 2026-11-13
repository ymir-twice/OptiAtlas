"""Versioned optical principles with deterministic, offline retrieval.

This package never interprets an image. Returned checks are guidance, not evidence.
It intentionally has no Qwen, network, embedding, or model dependency.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import re
import unicodedata

KNOWLEDGE_FILE = "optics_52_v1.json"
MAX_RESULTS = 5
MAX_QUERY_LENGTH = 1000
_STOPWORDS = frozenset("a an and are as at be by can for from has in is it of on or that the their this to with image visible light optics optical photograph photo".split())


def validate_lookup_args(category_ids: list[str], query: str = "", max_results: int = 3) -> None:
    """Reject malformed or unbounded requests; never coerce JSON scalar types."""
    if not isinstance(category_ids, list) or len(category_ids) > MAX_RESULTS:
        raise ValueError("category_ids must be a list with at most 5 category IDs")
    if any(not isinstance(cid, str) or not cid or len(cid) > 100 for cid in category_ids):
        raise ValueError("category_ids must contain nonempty strings of at most 100 characters")
    if len(set(category_ids)) != len(category_ids):
        raise ValueError("category_ids must not contain duplicates")
    if not isinstance(query, str) or len(query) > MAX_QUERY_LENGTH:
        raise ValueError("query must be a string of at most 1000 characters")
    if type(max_results) is not int or not 1 <= max_results <= MAX_RESULTS:
        raise ValueError("max_results must be an integer from 1 to 5")


def _tokens(value: str) -> set[str]:
    value = unicodedata.normalize("NFKC", value).casefold()
    result = set(re.findall(r"[a-z0-9]+", value)) - _STOPWORDS
    # Chinese bigrams enable short subject queries without a tokenizer dependency.
    for run in re.findall(r"[\u3400-\u9fff]+", value):
        if len(run) == 1:
            result.add(run)
        else:
            result.update(run[i:i + 2] for i in range(len(run) - 1))
    return result


@lru_cache(maxsize=1)
def _load() -> tuple[dict, str]:
    data = json.loads(files(__package__).joinpath(KNOWLEDGE_FILE).read_text(encoding="utf-8"))
    cards = data["cards"]
    ids = [card["category_id"] for card in cards]
    if data.get("schema_version") != 1 or len(ids) != 52 or len(set(ids)) != 52:
        raise ValueError("knowledge artifact must contain schema 1 and 52 unique category IDs")
    if data.get("evidence_role") != "conditional_guidance_not_image_evidence":
        raise ValueError("knowledge artifact must explicitly distinguish guidance from evidence")
    sources = {source["source_id"] for source in data["sources"]}
    if any(not card["source_ids"] or set(card["source_ids"]) - sources for card in cards):
        raise ValueError("knowledge artifact has missing source references")
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return data, "sha256:" + hashlib.sha256(canonical).hexdigest()


def load_knowledge() -> dict:
    """Return an isolated copy of all cards for offline auditing and authoring."""
    data, content_hash = _load()
    return {**deepcopy(data), "content_hash": content_hash}


def knowledge_lookup(category_ids: list[str], query: str = "", max_results: int = 3) -> dict:
    """Look up explicit IDs in order, then fill unused slots with lexical matches.

    Empty IDs and an empty query return the 52-item category catalog without cards.
    Unknown IDs fail explicitly. Query matches never imply image classification.
    """
    validate_lookup_args(category_ids, query, max_results)
    data, content_hash = _load()
    by_id = {card["category_id"]: card for card in data["cards"]}
    unknown = [cid for cid in category_ids if cid not in by_id]
    if unknown:
        raise ValueError("unknown category IDs: " + ", ".join(unknown))
    selected = list(category_ids[:max_results])
    terms = _tokens(query)
    ranked = []
    if terms:
        for card in data["cards"]:
            cid = card["category_id"]
            if cid in category_ids:
                continue
            title = " ".join((cid.replace("_", " "), card["name_en"], card["name_zh"]))
            examples = " ".join(card["search_queries"])
            body = " ".join(part["when"] + " " + part["visible_result"] for part in card["conditions_and_visible_results"])
            score = 5 * len(terms & _tokens(title)) + 3 * len(terms & _tokens(examples)) + len(terms & _tokens(body))
            if score:
                ranked.append((-score, cid))
        # Lexical category ID breaks equal-score ties independently of hash seed.
        ranked.sort()
        selected.extend(cid for _, cid in ranked[:max_results - len(selected)])
    chosen = [by_id[cid] for cid in selected]
    source_ids = {source_id for card in chosen for source_id in card["source_ids"]}
    result = {
        "status": "ok",
        "knowledge_version": data["knowledge_version"],
        "content_hash": content_hash,
        "evidence_role": data["evidence_role"],
        "usage_note": data["usage_note"],
        "selection_method": "explicit_ids_then_local_lexical_search",
        "requested_category_ids": list(category_ids),
        "unreturned_category_ids": list(category_ids[max_results:]),
        "query": query,
        "max_results": max_results,
        "total_categories": len(by_id),
        "cards": deepcopy(chosen),
        "sources": deepcopy([source for source in data["sources"] if source["source_id"] in source_ids]),
    }
    if not category_ids and not query.strip():
        result["category_catalog"] = [
            {key: card[key] for key in ("category_id", "name_en", "name_zh")}
            for card in data["cards"]
        ]
    return result


__all__ = ["knowledge_lookup", "load_knowledge", "validate_lookup_args"]
