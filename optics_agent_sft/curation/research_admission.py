"""Replay archived research tools without downloading anything or executing tools."""
from __future__ import annotations

from ..training.template import normalize_messages


def validate_research_arguments(name, args):
    if not isinstance(args, dict):
        raise ValueError("Research tool arguments must be an object")
    if name == "knowledge_lookup":
        if set(args) - {"category_ids", "query", "max_results"} or "category_ids" not in args:
            raise ValueError("Unsupported knowledge_lookup argument")
        ids = args.get("category_ids", [])
        query = args.get("query", "")
        if (not isinstance(ids, list) or len(ids) > 5
                or any(not isinstance(key, str) or not key.strip() or len(key) > 100 for key in ids)
                or len(set(ids)) != len(ids)
                or not isinstance(query, str) or len(query) > 1000):
            raise ValueError("knowledge_lookup requires valid category IDs and a bounded query")
        limit = args.get("max_results", 3)
        maximum = 5
    elif name == "image_search":
        if set(args) - {"query", "max_results"} or not isinstance(args.get("query"), str) or not 1 <= len(args["query"].strip()) <= 500:
            raise ValueError("image_search requires a bounded query")
        limit, maximum = args.get("max_results", 2), 2
    else:
        raise ValueError("Unknown research tool")
    if type(limit) is not int or not 1 <= limit <= maximum:
        raise ValueError("Invalid research result limit")


def validate_research_result(obj, step, native, ids, cursor, call_id, name, resolve, artifacts, seen):
    from .admission import strict_json_loads, read_json_snapshot
    from .research_protocol import REFERENCE_IMAGE_OBSERVATION_PROMPT

    result, messages = step.get("result", {}), step.get("result_messages", [])
    if result.get("status") != "ok" or result.get("tool_name") != name:
        raise ValueError("Research step must archive a completed matching tool execution")
    if (not isinstance(messages, list) or not 1 <= len(messages) <= 2
            or messages != native[cursor + 1:cursor + 1 + len(messages)]
            or step.get("result_message_ids") != ids[cursor + 1:cursor + 1 + len(messages)]):
        raise ValueError("Research result must match its exact causal message slice")
    tool = messages[0]
    if (set(tool) != {"role", "tool_call_id", "name", "content"}
            or tool["role"] != "tool" or tool["tool_call_id"] != call_id or tool["name"] != name
            or not isinstance(tool["content"], str)):
        raise ValueError("Research result has an invalid native call identity")
    payload = strict_json_loads(tool["content"])
    if not isinstance(payload, dict) or payload.get("tool") != name or payload.get("status") not in {"ok", "empty", "error"}:
        raise ValueError("Research payload requires its tool identity and explicit status")
    if result.get("tool_status") != payload["status"]:
        raise ValueError("Research tool status differs between message and step")
    validate_runtime_state(payload.get("runtime_state"), obj, step)
    args = strict_json_loads(step["completion"][0]["tool_calls"][0]["function"]["arguments"])
    if name == "image_search" and payload.get("query") != args["query"].strip():
        raise ValueError("Reference search query differs from the requested action")
    record = payload.get("record", {})
    if not isinstance(record, dict):
        raise ValueError("Research record must be an object")
    path = record.get("path")
    if (set(record) != {"path", "sha256"} or path not in artifacts or path in seen
            or artifacts[path].get("kind") != "research_result"
            or artifacts[path].get("tool_name") != name
            or artifacts[path].get("source_step_id") != step["step_id"]
            or record["sha256"] != artifacts[path]["sha256"]
            or result.get("result_path") != path or result.get("result_sha256") != record["sha256"]):
        raise ValueError("Research record is not a unique, matching, hashed tool artifact")
    archived, digest = read_json_snapshot(resolve(path))
    if digest != record["sha256"] or archived != {key: value for key, value in payload.items() if key != "record"}:
        raise ValueError("Research message differs from its immutable result snapshot")
    seen.add(path)
    references = payload.get("references", [])
    if not isinstance(references, list) or (name == "knowledge_lookup" and references):
        raise ValueError("Invalid reference image list for research tool")
    if name == "image_search" and len(references) > args.get("max_results", 2):
        raise ValueError("Reference image count exceeds the requested result limit")
    paths = []
    for ref in references:
        if not isinstance(ref, dict):
            raise ValueError("Reference image entry must be an object")
        image = ref.get("image_path")
        if (image not in artifacts or image in seen or artifacts[image].get("kind") != "reference_image"
                or artifacts[image].get("source_step_id") != step["step_id"]
                or ref.get("sha256") != artifacts[image]["sha256"]):
            raise ValueError("Reference image must be a newly returned, independently hashed reference artifact")
        expected_source = {key: ref.get(key) for key in ("source_url", "landing_page", "provider", "source_id")}
        expected_source.update({key: ref[key] for key in (
            "creator", "attribution", "license", "license_url", "license_unknown",
            "license_evidence", "photographic_origin_verified", "role") if key in ref})
        if artifacts[image].get("provenance") != expected_source:
            raise ValueError("Reference artifact provenance differs from the returned search result")
        resolve(image)
        seen.add(image)
        paths.append(image)
    if paths:
        expected = {"role": "user", "content": [{"type": "text", "text": REFERENCE_IMAGE_OBSERVATION_PROMPT}]
                    + [{"type": "image_url", "image_url": {"url": image}} for image in paths]}
        if len(messages) != 2 or messages[1] != expected:
            raise ValueError("Reference observation must contain exactly the retrieved pixels and published instruction")
    elif len(messages) != 1:
        raise ValueError("A research result without images must not introduce an observation")
    if name == "image_search" and result.get("reference_image_paths") != paths:
        raise ValueError("Reference paths disagree between the step and observation")
    normalize_messages(messages)
    return 1 + len(messages)


def validate_runtime_state(state, obj, step, *, draw=False):
    """Recompute budgets from causal actions, not teacher claims or final counters."""
    prefix = obj["steps"][:obj["steps"].index(step) + 1]
    calls = [s["completion"][0].get("tool_calls", []) for s in prefix]
    names = [call[0]["function"]["name"] for call in calls if call]
    knowledge, search, images = (names.count(name) for name in ("knowledge_lookup", "image_search", "draw"))
    config = obj["config"]
    limits = ("max_knowledge_calls", "max_search_calls", "max_image_calls", "max_provider_calls", "max_steps")
    if any(type(config.get(key)) is not int or config[key] < 0 for key in limits):
        raise ValueError("Research runtime requires explicit nonnegative integer budgets")
    expected = {"knowledge_calls_used": knowledge, "remaining_knowledge_calls": config["max_knowledge_calls"] - knowledge,
                "search_calls_used": search, "remaining_search_calls": config["max_search_calls"] - search,
                "remaining_image_calls": config["max_image_calls"] - images,
                "remaining_provider_calls": config["max_provider_calls"] - len(prefix) - images - search,
                "remaining_steps": config["max_steps"] - len(prefix), "generated_image_available": images > 0}
    fields = set(expected)
    if draw:
        fields |= {"schema_version", "image_calls_used", "max_image_calls", "draw_available",
                   "identical_prompt_to_previous", "identical_image_to_previous", "instruction"}
    if not isinstance(state, dict) or set(state) != fields:
        raise ValueError("Research runtime state differs from the published fields")
    for key, value in expected.items():
        if type(state[key]) is not type(value) or state[key] != value or (type(value) is int and value < 0):
            raise ValueError("Research runtime budget disagrees with the replayed actions")
