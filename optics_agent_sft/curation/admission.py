"""Fail-closed trajectory admission, provenance and scene-family isolation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..training.template import (PROJECTION_VERSION, normalize_messages,
                                 reject_reserved_markup, validate_transport_metadata)

TOOLS = {"draw"}
# Published runtime protocols. New legitimate instructions must be
# registered explicitly; otherwise hidden evaluation text could be smuggled
# into the system/tool/observation fields despite an updated review-file hash.
SUPPORTED_SYSTEM_SHA256 = {
    "9ec07c2289137e1e87c599d2b0ddd9be2ea7d996f329191766072b2711c305a1",  # Native research tools, optional references and original ReAct draw loop.
    "b5ff9db47d329608ae734a84bc010d60486a25d16704081f775294c4b6ddb3fc",  # Runner runtime_state controls actions, not optical acceptance.
    "96e120b109bb11531f3e4bd4ded99fa0dc731cb51ac06c6ae7369f42015dfe5e",  # ReAct with optional public review and strategy changes after repeated failure.
    "4bb0d3feca26ce0e942a22d4f27e475e8b0e2dcb25148f0cc390195ba1301c8b",  # ReAct loop with five-section draw action interface.
    "09708d5b810e91dbccc1bd3cf978b4acd3ce7016267c62d7a1fe33e699a29a45",  # Original draw protocol.
    "82609346fa7c286050bde76fa23db0c1ba62f16da188f07adf0d75a088ccf867",  # optics_visible_results_v2.
    "6c9cb865730db553f18eea55ba857fc3b81f28b1c892d251ba06eb344efe8a6e",  # Five sections with two-dimensional composition.
}
SUPPORTED_TOOLS_SHA256 = {
    "c091283c3abaca4237b8dc17b23d0e0fc060a16f3cebed545d7cf3f1a3a195aa",  # Native draw, knowledge_lookup and Pexels image_search.
    "035f02bafa07b404ba94aea4bd80f61206526e80ae1eaf1ed0267fd4243669ad",  # ReAct draw interface, qualitative 3D planning.
    "5c81b0d191591524345e09e9470928908fa0876e13e050c96e44315109b07077",  # Original draw description.
    "9f4eeb968944c0f363f97c1a238e59e44798fc4976cc55cb4c6ca0c2a442fc6e",  # Six-section generator caption.
    "6a3f81da099aeb0e74d27235025a59de87faf4712078e76e71d4d1ac9ce60cc2",  # Five-section generator caption.
}
DRAW_OBSERVATION_TEXT = ("Image returned by draw. Inspect these actual pixels against the original request "
                         "before deciding whether to draw again or submit.")
# Frozen public ReAct observation. Keep admission independent of the runtime install.
REACT_DRAW_OBSERVATION_TEXT = 'Image returned by draw. OBSERVE and EVALUATE these actual pixels against the\noriginal user request, not merely the previous draw prompt. Check hard constraints,\nthe relevant visible optical relationships, and whether the evidence is clear\nenough to inspect. Distinguish satisfied, visibly contradicted and not verifiable\nrequirements; do not infer exact 3D geometry or hidden parameters from 2D projection\nalone. Provide a brief image-grounded review, then either redraw with a complete\ncorrected caption preserving the original constraints, or submit the latest image\nif ready. Keep the reasoning trace out of the caption and final response.'
OPTIONAL_REVIEW_DRAW_OBSERVATION_TEXT = 'Image returned by draw. OBSERVE and EVALUATE these actual pixels against the\noriginal user request, not merely the previous draw prompt. Check hard constraints,\nthe relevant visible optical relationships, and whether the evidence is clear\nenough to inspect. Distinguish satisfied, visibly contradicted and not verifiable\nrequirements; do not infer exact 3D geometry or hidden parameters from 2D projection\nalone. Review internally; public commentary is optional and should be minimal.\nThen either redraw with a complete\ncorrected caption preserving the original constraints, or submit the latest image\nif ready. Keep the reasoning trace out of the caption and final response.'
SUPPORTED_DRAW_OBSERVATIONS = {
    DRAW_OBSERVATION_TEXT, REACT_DRAW_OBSERVATION_TEXT, OPTIONAL_REVIEW_DRAW_OBSERVATION_TEXT,
}
MAX_JSON_BYTES = 64 * 1024 * 1024


def strict_json_loads(text: str):
    """Reject ambiguous duplicate keys/nonfinite values before any projection."""
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result
    def constant(_value):
        raise ValueError("Nonfinite JSON values are not supported")
    try:
        return json.loads(text, object_pairs_hook=object_pairs, parse_constant=constant)
    except RecursionError as error:
        raise ValueError("JSON nesting is too deep") from error


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_hash(prompt: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def isolation_key(value: str) -> str:
    """Canonical IDs for overlap checks only; retain original IDs in samples."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Reserved/group identity must be a nonempty string")
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def read_json_snapshot(path: Path) -> tuple[dict, str]:
    """Parse and identify the same bounded byte snapshot, never a later file."""
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("JSON file exceeds the 64 MiB admission limit")
    return strict_json_loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    return read_json_snapshot(path)[0]


def local_artifact(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value or value.startswith("data:"):
        raise ValueError("Artifact must be a local path")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes trajectory directory: {value}")
    if not path.is_file():
        raise ValueError(f"Missing artifact: {value}")
    return path


@dataclass
class Admission:
    samples: list[dict]
    accepted: list[dict]
    rejected: list[dict]

    def summary(self) -> dict:
        return {"accepted_trajectories": len(self.accepted), "decision_samples": len(self.samples),
                "train_samples": sum(s["split"] == "train" for s in self.samples),
                "validation_samples": sum(s["split"] == "validation" for s in self.samples),
                "demo_samples": sum(s["is_demo"] for s in self.samples),
                "contains_demo": any(s["is_demo"] for s in self.samples),
                "accepted": self.accepted, "rejected": self.rejected}


def trajectory_paths(roots: list[str | Path]) -> list[Path]:
    paths = set()
    for raw in roots:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            paths.update(path.rglob("trajectory.json"))
        else:
            raise ValueError(f"Trajectory path does not exist: {path}")
    if not paths:
        raise ValueError("No trajectory.json files found")
    return sorted(paths)


def reservations(extra_paths: list[str | Path] | None = None) -> dict[str, set]:
    result = {k: set() for k in ("scene_families", "task_ids", "prompt_sha256", "image_sha256")}
    paths = [Path(__file__).parent.parent / "benchmark_exclusions.json", *map(Path, extra_paths or [])]
    for path in paths:
        obj = read_json(path)
        # Saved data-collection exclusions predate the versioned SFT envelope.
        # Accept only that explicit flat format; keep validating every reserved
        # ID/hash below. Never ignore unsupported source-level exclusion rules.
        legacy_fields = {"task_ids", "prompt_sha256", "image_sha256", "source_identities", "inputs"}
        legacy = (isinstance(obj, dict) and "schema_version" not in obj
                  and set(obj) == legacy_fields and obj["source_identities"] == []
                  and isinstance(obj["inputs"], list))
        if not isinstance(obj, dict) or (obj.get("schema_version") != 1 and not legacy):
            raise ValueError(f"Unsupported reservation schema: {path}")
        for key in result:
            if not isinstance(obj.get(key, []), list):
                raise ValueError(f"Reservation {key} must be a list")
            values = obj.get(key, [])
            if key in {"scene_families", "task_ids"}:
                values = [isolation_key(value) for value in values]
            elif any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in values):
                raise ValueError("Reserved hashes must be lowercase SHA256 strings")
            result[key].update(values)
    return result


def _identity(trajectory: dict) -> dict:
    task = trajectory.get("task", {})
    for key in ("task_id", "scene_family", "prompt", "split"):
        if not isinstance(task.get(key), str) or not task[key].strip():
            raise ValueError(f"task.{key} must be a nonempty string")
    artifacts = trajectory.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("artifacts must be a list")
    original_split = task["split"].strip().lower()
    effective_split = ("demo" if trajectory.get("is_demo") is True
                       and original_split in {"train", "validation", "demo"} else original_split)
    return {"scene_families": isolation_key(task["scene_family"]), "task_ids": isolation_key(task["task_id"]),
            "prompt_sha256": prompt_hash(task["prompt"]),
            # Public deterministic theory diagrams may be shared across splits.
            # Generated/edit/crop image bytes remain strictly isolated.
            "image_sha256": {a.get("sha256") for a in artifacts if isinstance(a, dict)
                             and a.get("sha256") and a.get("kind") not in {"reference", "reference_image", "research_result"}},
            "all_image_sha256": {a.get("sha256") for a in artifacts if isinstance(a, dict)
                                 and a.get("sha256") and a.get("kind") != "research_result"},
            "split": effective_split, "original_split": original_split}


def _strip_id(message: dict) -> dict:
    return {key: value for key, value in message.items() if key != "message_id"}


def _validate_assistant(message: dict, tools: list[dict]) -> bool:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("Selected completion must be one native assistant message")
    if set(message) - {"role", "content", "tool_calls"}:
        raise ValueError("Unsupported assistant fields; private reasoning/custom decision JSON is not an SFT target")
    if not isinstance(message.get("content", ""), (str, type(None))) or len(message.get("content") or "") > 6000:
        raise ValueError("Assistant content must be bounded public text")
    normalize_messages([message])  # Validate raw delimiters, excluding opaque transport signatures.
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list):
        raise ValueError("Native tool_calls must be a list when present")
    if calls:
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("Draw-only runtime requires exactly one tool call per assistant turn")
        call = calls[0]
        if set(call) - {"id", "type", "function", "extra_content"}:
            raise ValueError("Unsupported native tool call fields")
        validate_transport_metadata(call)
        if call.get("type") != "function" or not isinstance(call.get("id"), str) or not 1 <= len(call["id"]) <= 200:
            raise ValueError("Native function call needs a nonempty id")
        function = call.get("function", {})
        allowed_tools = {tool["function"]["name"] for tool in tools}
        if set(function) != {"name", "arguments"} or function.get("name") not in allowed_tools:
            raise ValueError("Only declared native tool calls are trainable")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ValueError("Native draw arguments must be a JSON string")
        if len(arguments) > 131072:
            raise ValueError("Native draw arguments exceed the bounded JSON text limit")
        arguments = strict_json_loads(arguments)
        if function["name"] != "draw":
            from .research_admission import validate_research_arguments
            validate_research_arguments(function["name"], arguments)
            reject_reserved_markup(arguments)
            return True
        if not isinstance(arguments, dict) or set(arguments) != {"prompt"}:
            raise ValueError("draw requires exactly one prompt argument")
        if not isinstance(arguments["prompt"], str) or not 1 <= len(arguments["prompt"].strip()) <= 10000:
            raise ValueError("draw.prompt must be nonempty and at most 10000 characters")
        reject_reserved_markup(arguments)
        return True
    if not isinstance(message.get("content"), str) or not message["content"].strip():
        raise ValueError("Final assistant response must contain public text")
    return False


def _validate_evaluation(final: dict, task: dict) -> None:
    evaluation = final.get("evaluation", {})
    pair = (final.get("status"), evaluation.get("status"))
    if pair == ("submitted", "not_run"):
        if set(evaluation) != {"status"}:
            raise ValueError("An unrun external judge cannot contain recorded judgments")
        return
    if pair != ("success", "passed"):
        raise ValueError("Final must be submitted/not_run or success/passed; rejected and technical failures are excluded")
    checks, global_check = evaluation.get("checks"), evaluation.get("global_consistency")
    if not isinstance(checks, list) or not checks or not isinstance(global_check, dict):
        raise ValueError("External pass requires nonempty checks and a global consistency judgment")
    for check in checks + [global_check]:
        if (not isinstance(check, dict) or check.get("answer") != "A"
                or not isinstance(check.get("evidence"), str) or not check["evidence"].strip()):
            raise ValueError("Every recorded judge check must be A with evidence; B/C cannot be overridden")
    ids = [check.get("id") for check in checks]
    if any(not isinstance(value, str) or not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("External judge check IDs must be nonempty and unique")
    expected_checks = task.get("checks", [])
    if not isinstance(expected_checks, list):
        raise ValueError("Task final-only checks must be a list")
    expected_ids = [check.get("id") if isinstance(check, dict) else None for check in expected_checks]
    if (any(not isinstance(value, str) or not value.strip() for value in expected_ids)
            or len(set(expected_ids)) != len(expected_ids)):
        raise ValueError("Task final-only check IDs must be nonempty and unique")
    expected = set(expected_ids)
    if expected and set(ids) != expected:
        raise ValueError("External judge check IDs do not match the task's final-only checks")


def _validate_draw_runtime_state(payload: dict, config: dict, *, draw_number: int,
                                 prompt: str, image_sha256: str,
                                 previous_prompt: str | None,
                                 previous_image_sha256: str | None) -> None:
    """Verify the opt-in runtime protocol without depending on optics_agent.

    Instruction strings are a frozen schema-v1 allowlist, not arbitrary tool
    commentary. Recompute state from the replayed calls and hashed artifacts so
    neither a reapproved JSON edit nor hidden evaluation text can redefine it.
    """
    state = payload.get("runtime_state")
    fields = {"schema_version", "image_calls_used", "max_image_calls", "remaining_image_calls",
              "draw_available", "identical_prompt_to_previous", "identical_image_to_previous",
              "instruction"}
    if not isinstance(state, dict) or set(state) != fields:
        raise ValueError("Draw runtime state must match the complete published schema")
    for field in ("schema_version", "image_calls_used", "max_image_calls", "remaining_image_calls"):
        if type(state[field]) is not int:
            raise ValueError("Draw runtime state counters must be integers, excluding bool")
    for field in ("draw_available", "identical_prompt_to_previous", "identical_image_to_previous"):
        if type(state[field]) is not bool:
            raise ValueError("Draw runtime state flags must be booleans")
    maximum = config.get("max_image_calls")
    if type(maximum) is not int or maximum <= 0 or draw_number > maximum:
        raise ValueError("Draw runtime state requires a positive configured draw budget and cannot exceed it")
    remaining = maximum - draw_number
    same_prompt = previous_prompt is not None and prompt == previous_prompt
    same_image = previous_image_sha256 is not None and image_sha256 == previous_image_sha256
    instruction = "Review the returned pixels against the original request. "
    if remaining:
        instruction += ("If a material defect remains, change the relevant caption constraints before another draw; "
                        "otherwise submit.")
    else:
        instruction += ("No draw calls remain. Give a final response stating any unresolved material defects; "
                        "do not request another draw.")
    if same_prompt and same_image:
        instruction += (" The caption and image are unchanged from the previous draw. "
                        "Repeating this action with the same seed will not make progress.")
    expected = {"schema_version": 1, "image_calls_used": draw_number, "max_image_calls": maximum,
                "remaining_image_calls": remaining, "draw_available": remaining > 0,
                "identical_prompt_to_previous": same_prompt, "identical_image_to_previous": same_image,
                "instruction": instruction}
    if state != expected:
        raise ValueError("Draw runtime state disagrees with the replayed calls, images or published instructions")


def _validate_transcript(obj: dict, tools: list[dict], resolve, artifact_specs: dict) -> dict:
    """Replay every archived turn, including unselected history, without tools.

    An approved target can only learn from images that its actual earlier draw
    returned. Equality of duplicated context text alone does not establish this.
    """
    messages, steps, final = obj["messages"], obj["steps"], obj["final"]
    ids = [message["message_id"] for message in messages]
    native = [_strip_id(message) for message in messages]
    if any(set(message) != {"role", "content"} for message in native[:2]):
        raise ValueError("Initial system/user messages cannot contain extra native fields")
    system = native[0]["content"]
    if not isinstance(system, str) or hashlib.sha256(system.encode("utf-8")).hexdigest() not in SUPPORTED_SYSTEM_SHA256:
        raise ValueError("System prompt does not match a published prompt-only runtime protocol")
    normalize_messages(native[:2])
    cursor, seen_calls, seen_images, latest_image, records = 2, set(), set(), None, {}
    config = obj.get("config", {})
    if isinstance(config, dict) and "expose_runtime_state" in config and type(config["expose_runtime_state"]) is not bool:
        raise ValueError("config.expose_runtime_state must be an explicit boolean")
    runtime_state_enabled = (isinstance(config, dict) and config.get("expose_runtime_state") is True
                             or obj.get("framework", {}).get("agent_protocol") == "native-optics-research-v1")
    previous_draw_prompt, previous_image_sha256 = None, None
    research_enabled = obj.get("framework", {}).get("agent_protocol") == "native-optics-research-v1"
    seen_research, draw_number = set(), 0
    for step in steps:
        context, completion = step.get("context"), step.get("completion")
        if (not isinstance(context, list) or not context or not isinstance(completion, list)
                or len(completion) != 1 or cursor >= len(native)
                or context != native[:cursor] or completion != native[cursor:cursor + 1]
                or step.get("context_message_ids") != ids[:cursor]
                or step.get("completion_message_ids") != ids[cursor:cursor + 1]):
            raise ValueError("Every step must match the next exact causal prefix of archived messages")
        is_call = _validate_assistant(completion[0], tools)
        records[step["step_id"]] = (context, completion, cursor)
        metadata = step.get("actor_metadata") or {}
        if "input_images" in metadata:
            expected_inputs = []
            for message in context:
                for part in message.get("content", []) if isinstance(message.get("content"), list) else []:
                    if part.get("type") == "image_url":
                        image = part["image_url"]["url"]
                        if image not in artifact_specs:
                            raise ValueError("Actor input image is absent from recorded artifacts")
                        expected_inputs.append({"path": image, "sha256": artifact_specs[image]["sha256"]})
            if metadata["input_images"] != expected_inputs:
                raise ValueError("Actual actor input image order/SHA256 disagrees with its causal context")
        result, result_messages = step.get("result", {}), step.get("result_messages", [])
        if not is_call:
            if (step is not steps[-1] or cursor != len(native) - 1 or latest_image is None
                    or result.get("status") != "submitted" or result_messages
                    or step.get("result_message_ids", [])
                    or final.get("assistant_message_id") != ids[cursor]):
                raise ValueError("Only the actual terminal assistant response may submit the latest image")
            cursor += 1
            continue
        call_id = completion[0]["tool_calls"][0]["id"]
        if call_id in seen_calls:
            raise ValueError("Native tool call IDs must be unique")
        seen_calls.add(call_id)
        tool_name = completion[0]["tool_calls"][0]["function"]["name"]
        if tool_name != "draw":
            if not research_enabled:
                raise ValueError("Research calls require the published research protocol")
            from .research_admission import validate_research_result
            cursor += validate_research_result(obj, step, native, ids, cursor, call_id, tool_name,
                                                resolve, artifact_specs, seen_research)
            continue
        draw_number += 1
        if result.get("status") != "ok":
            raise ValueError("A successful draw-only transcript cannot contain a failed execution")
        if (not isinstance(result_messages, list) or len(result_messages) != 2
                or result_messages != native[cursor + 1:cursor + 3]
                or ("result_message_ids" in step and step["result_message_ids"] != ids[cursor + 1:cursor + 3])):
            raise ValueError("Draw must archive exactly its matching tool result and one following image observation")
        tool, observation = result_messages
        if (set(tool) != {"role", "tool_call_id", "name", "content"} or tool.get("role") != "tool"
                or tool.get("tool_call_id") != call_id or tool.get("name") != "draw"
                or not isinstance(tool.get("content"), str)):
            raise ValueError("Draw native tool result has an invalid call identity or fields")
        payload = strict_json_loads(tool["content"])
        allowed_payload_fields = {"status", "image_path", "sha256", "width", "height"}
        if runtime_state_enabled:
            allowed_payload_fields.add("runtime_state")
        if (not isinstance(payload, dict) or payload.get("status") != "ok"
                or set(payload) - allowed_payload_fields):
            raise ValueError("Draw native tool result must describe only its successful image")
        image_path = payload.get("image_path")
        if not isinstance(image_path, str) or image_path not in artifact_specs or image_path in seen_images:
            raise ValueError("Each draw must return its own uniquely recorded image artifact")
        artifact = artifact_specs[image_path]
        if artifact.get("kind") != "generated_image":
            raise ValueError("Draw must return generated pixels, not reference images or tool records")
        if payload.get("sha256") != artifact["sha256"] or result.get("image_path") != image_path:
            raise ValueError("Draw result image path/SHA256 disagrees with its artifact or step result")
        for dimension in ("width", "height"):
            if dimension in payload or dimension in artifact:
                value = payload.get(dimension)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value != artifact.get(dimension):
                    raise ValueError("Draw dimensions disagree with its image artifact")
        draw_prompt = strict_json_loads(completion[0]["tool_calls"][0]["function"]["arguments"])["prompt"]
        if runtime_state_enabled:
            runtime_payload = payload
            if research_enabled:
                from .research_admission import validate_runtime_state
                validate_runtime_state(payload.get("runtime_state"), obj, step, draw=True)
                draw_fields = {"schema_version", "image_calls_used", "max_image_calls", "remaining_image_calls",
                               "draw_available", "identical_prompt_to_previous", "identical_image_to_previous", "instruction"}
                runtime_payload = {**payload, "runtime_state": {key: payload["runtime_state"][key] for key in draw_fields}}
            _validate_draw_runtime_state(
                runtime_payload, config, draw_number=draw_number, prompt=draw_prompt,
                image_sha256=artifact["sha256"], previous_prompt=previous_draw_prompt,
                previous_image_sha256=previous_image_sha256,
            )
        expected_observations = [{"role": "user", "content": [
            {"type": "text", "text": review_text},
            {"type": "image_url", "image_url": {"url": image_path}}]}
            for review_text in SUPPORTED_DRAW_OBSERVATIONS]
        if observation not in expected_observations:
            raise ValueError("Draw image observation does not match the actual returned image and public runtime text")
        normalize_messages(result_messages)
        resolve(image_path)
        seen_images.add(image_path)
        latest_image = image_path
        previous_draw_prompt, previous_image_sha256 = draw_prompt, artifact["sha256"]
        cursor += 3
    if cursor != len(native) or latest_image != final.get("image_path") or seen_images | seen_research != set(artifact_specs):
        raise ValueError("Transcript has extra messages/artifacts or final image is not the last returned image")
    return records


def _admit(path: Path, obj: dict, allow_demo: bool, *, trajectory_hash: str,
           artifact_map: dict[str, Path] | None = None,
           accepted_review_types: tuple[str, ...] = ("human",)) -> tuple[list[dict], dict]:
    if obj.get("schema_version") != 2:
        raise ValueError("Only native trajectory schema_version 2 is accepted; legacy schema 1 requires a new run and review")
    if obj.get("framework", {}).get("name") != "qwen-agent":
        raise ValueError("Expected qwen-agent framework provenance")
    if obj["framework"].get("message_projection") != PROJECTION_VERSION:
        raise ValueError("Runtime/SFT native message projection version mismatch")
    if not isinstance(obj.get("is_demo"), bool):
        raise ValueError("is_demo must be explicit")
    is_demo = obj["is_demo"]
    if is_demo and not allow_demo:
        raise ValueError("Demo trajectories are excluded")
    ident = _identity(obj)
    if ident["split"] not in {"train", "validation"} and not (is_demo and allow_demo and ident["split"] == "demo"):
        raise ValueError("Benchmark/test splits never enter SFT")
    tools = obj.get("tools")
    research_enabled = obj.get("framework", {}).get("agent_protocol") == "native-optics-research-v1"
    names = [tool.get("function", {}).get("name") for tool in tools] if isinstance(tools, list) else []
    if (not names or any(tool.get("type") != "function" for tool in tools)
            or names != (["draw", "knowledge_lookup", "image_search"] if research_enabled else ["draw"])):
        raise ValueError("Native tools must match the declared draw or research protocol")
    parameters = tools[0]["function"].get("parameters", {})
    if parameters.get("type") != "object" or set(parameters.get("properties", {})) != {"prompt"} or parameters.get("required") != ["prompt"]:
        raise ValueError("draw schema must require exactly prompt")
    tools_hash = hashlib.sha256(json.dumps(tools, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    if tools_hash not in SUPPORTED_TOOLS_SHA256:
        raise ValueError("Tool definitions do not match a published prompt-only draw protocol")
    steps = obj.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("No trajectory steps")
    step_ids = [s.get("step_id") for s in steps]
    if any(not isinstance(s, str) or not s for s in step_ids) or len(step_ids) != len(set(step_ids)):
        raise ValueError("step_id must be unique strings")
    if not accepted_review_types or set(accepted_review_types) - {"human", "model"}:
        raise ValueError("accepted_review_types must explicitly select human and/or model")
    review_kind = "human"
    review_path = path.with_name("human_review.json")
    if not is_demo and ("human" not in accepted_review_types or not review_path.is_file()) and "model" in accepted_review_types:
        review_kind = "model"
        review_path = path.with_name("machine_review.json")
    review, review_hash = None, None
    if not is_demo:
        if not review_path.is_file():
            raise ValueError(f"Missing {review_path.name}")
        review, review_hash = read_json_snapshot(review_path)
        if review_kind == "model":
            judgement = review.get("judgement", {})
            judge = review.get("judge", {})
            if (not isinstance(judgement, dict) or not isinstance(judge, dict)
                    or review.get("review_type") != "model" or judgement.get("assessable") is not True
                    or not isinstance(judge.get("model"), str) or not judge["model"].strip()
                    or not re.fullmatch(r"[0-9a-f]{64}", str(judge.get("prompt_sha256", "")))):
                raise ValueError("Machine review needs explicit model identity, prompt hash and assessable evidence")
            thresholds = review.get("thresholds")
            if not isinstance(thresholds, dict) or set(thresholds) != {"optical", "prompt"}:
                raise ValueError("Machine review needs recorded optical and prompt thresholds")
            for score, threshold_name in (("optical_score", "optical"), ("prompt_score", "prompt")):
                value = judgement.get(score)
                threshold = thresholds[threshold_name]
                if (isinstance(value, bool) or not isinstance(value, (float, int))
                        or not 0 <= value <= 1 or not math.isfinite(value)):
                    raise ValueError("Invalid machine review score")
                if (isinstance(threshold, bool) or not isinstance(threshold, (float, int))
                        or not 0 <= threshold <= 1 or not math.isfinite(threshold)):
                    raise ValueError("Invalid machine review threshold")
                if value < threshold:
                    raise ValueError("Machine review score is below its recorded threshold")
        if review.get("schema_version") != 2 or review.get("decision") != "approved":
            raise ValueError("Native schema 2 needs an explicit schema 2 human approval")
        if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            raise ValueError("Human reviewer identity is required")
        reviewed_at = datetime.fromisoformat(str(review.get("reviewed_at", "")).replace("Z", "+00:00"))
        if reviewed_at.tzinfo is None:
            raise ValueError("Review timestamp must include timezone")
        if review.get("final_approved") is not True or review.get("trajectory_sha256") != trajectory_hash:
            raise ValueError("Final approval missing or trajectory SHA256 changed")
        for key in ("optical_validity", "prompt_fulfillment"):
            evidence = review.get("final_review", {}).get(key)
            if not isinstance(evidence, str) or len(evidence.strip()) < 12:
                raise ValueError("Human final_review needs optical_validity and prompt_fulfillment evidence")
        selected = review.get("selected_step_ids")
        if (not isinstance(selected, list) or not selected or any(not isinstance(s, str) for s in selected)
                or len(selected) != len(set(selected)) or not set(selected).issubset(step_ids)):
            raise ValueError("selected_step_ids must select unique existing steps")
    else:
        selected = step_ids
    final = obj.get("final", {})
    _validate_evaluation(final, obj["task"])
    root = path.parent
    def resolve(value):
        if (not isinstance(value, str) or not value or Path(value).is_absolute()
                or ".." in Path(value).parts or "://" in value or value.startswith("data:")):
            raise ValueError("Raw trajectory artifacts must use local relative paths inside the run")
        if artifact_map is None:
            return local_artifact(root, value)
        if not isinstance(value, str) or value not in artifact_map:
            raise ValueError("Portable source artifact is absent from its manifest mapping")
        return artifact_map[value]
    artifact_hashes, artifact_paths, artifact_specs = {}, {}, {}
    for artifact in obj.get("artifacts", []):
        resolved = resolve(artifact.get("path"))
        if artifact["path"] in artifact_specs:
            raise ValueError("Duplicate artifact path")
        if str(resolved) in artifact_hashes and artifact_hashes[str(resolved)] != artifact.get("sha256"):
            raise ValueError("Conflicting artifact path")
        if artifact.get("kind") not in ({"generated_image", "reference_image", "research_result"} if research_enabled else {"generated_image"}):
            raise ValueError("Schema 2 draw-only trajectories use generated_image artifacts")
        expected = artifact.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(resolved) != expected:
            raise ValueError("Artifact SHA256 mismatch")
        artifact_hashes[str(resolved)] = expected
        artifact_paths[artifact["path"]] = str(resolved)
        artifact_specs[artifact["path"]] = artifact
    final_path = resolve(final.get("image_path"))
    if artifact_specs.get(final.get("image_path"), {}).get("kind") != "generated_image":
        raise ValueError("Final image must be a generated image, never a retrieved reference")
    if str(final_path) not in artifact_hashes or final.get("image_sha256") != artifact_hashes[str(final_path)]:
        raise ValueError("Final full image does not match hashed artifacts")
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Native trajectory must preserve messages")
    if (len(messages) < 2 or messages[0].get("role") != "system"
            or messages[1].get("role") != "user"
            or not isinstance(messages[1].get("content"), str)
            or messages[1]["content"] != obj["task"]["prompt"]):
        raise ValueError("Initial native messages must be system + plain-text user containing exactly task.prompt")
    ids = [message.get("message_id") for message in messages]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Global native message IDs must be unique")
    if final.get("assistant_message_id") != ids[-1] or messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls"):
        raise ValueError("Final submission must identify the last ordinary assistant message")
    records = _validate_transcript(obj, tools, resolve, artifact_specs)
    samples = []
    for step in steps:
        if step["step_id"] not in selected:
            continue
        context, completion, n = records[step["step_id"]]
        conversation = copy.deepcopy(context + completion)
        ordered_hashes = []
        for message in conversation:
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError("Unsupported native role")
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        value = part.get("image_url")
                        value = value.get("url") if isinstance(value, dict) else value
                        resolved = resolve(value)
                        if str(resolved) not in artifact_hashes:
                            raise ValueError("Context image missing from hashed artifacts")
                        part["image_url"] = {"url": str(resolved)}
                        ordered_hashes.append(artifact_hashes[str(resolved)])
                    elif part.get("type") != "text" or not isinstance(part.get("text"), str):
                        raise ValueError("Only native text/image_url parts are accepted")
        samples.append({"schema_version": 2, "sample_id": f"{obj['trajectory_id']}:{step['step_id']}",
                        "trajectory_sha256": trajectory_hash, "step_id": step["step_id"],
                        "scene_family": obj["task"]["scene_family"], "task_id": obj["task"]["task_id"],
                        "prompt_sha256": ident["prompt_sha256"], "split": ident["split"],
                        "original_split": ident["original_split"], "is_demo": is_demo,
                        "tools": copy.deepcopy(tools), "messages": conversation,
                        "target_message_index": n, "image_sha256": ordered_hashes,
                        "projection_version": PROJECTION_VERSION})
    if sha256_file(path) != trajectory_hash or (review_hash and sha256_file(review_path) != review_hash):
        raise ValueError("Trajectory or human review changed during admission")
    metadata = {"trajectory_path": str(path), "trajectory_sha256": trajectory_hash,
                "human_review_sha256": review_hash if review_kind == "human" else None,
                "machine_review_sha256": review_hash if review_kind == "model" else None,
                "review_type": review_kind if review else None,
                "reviewer": review["reviewer"] if review else None,
                "selected_step_ids": selected, "scene_family": obj["task"]["scene_family"],
                "split": ident["split"], "original_split": ident["original_split"],
                "is_demo": is_demo, "models": obj.get("models", {}), "framework": obj.get("framework"),
                "artifact_sha256": artifact_hashes, "artifact_paths": artifact_paths}
    return samples, metadata


def admit_trajectories(roots: list[str | Path], *, reserved_paths: list[str | Path] | None = None,
                       allow_demo: bool = False, artifact_maps: dict | None = None,
                       accepted_review_types: tuple[str, ...] = ("human",)) -> Admission:
    reserved = reservations(reserved_paths)
    loaded, rejected = [], []
    identities = {k: {} for k in ("scene_families", "task_ids", "prompt_sha256", "image_sha256")}
    for path in trajectory_paths(roots):
        try:
            obj, trajectory_hash = read_json_snapshot(path)
            ident = _identity(obj)
            loaded.append((path, obj, ident, trajectory_hash))
            for key in identities:
                values = ident[key] if isinstance(ident[key], set) else {ident[key]}
                for value in values:
                    identities[key].setdefault(value, set()).add(ident["split"])
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
            rejected.append({"trajectory_path": str(path), "reason": str(error)})
    samples, accepted = [], []
    for path, obj, ident, trajectory_hash in loaded:
        try:
            for key in identities:
                values = ident[key] if isinstance(ident[key], set) else {ident[key]}
                reserved_values = ident["all_image_sha256"] if key == "image_sha256" else values
                if reserved_values & reserved[key]:
                    raise ValueError(f"Reserved benchmark/test overlap: {key}")
                if any(len(identities[key][value]) > 1 for value in values):
                    raise ValueError(f"Cross-split leakage: {key}")
            rows, metadata = _admit(path, obj, allow_demo,
                                    trajectory_hash=trajectory_hash,
                                    artifact_map=(artifact_maps or {}).get(str(path)),
                                    accepted_review_types=accepted_review_types)
            if any(s["sample_id"] in {a["sample_id"] for a in samples} for s in rows):
                raise ValueError("Duplicate decision sample_id")
            samples.extend(rows)
            accepted.append(metadata)
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
            rejected.append({"trajectory_path": str(path), "reason": str(error)})
    return Admission(samples, accepted, rejected)
