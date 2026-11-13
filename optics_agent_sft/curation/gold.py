"""Verified, selected gold actions, including useful nonterminal draw prefixes.

This is a separate admission format. Ordinary native trajectories keep their
existing final-submission gate. Judge evidence and rejected candidates remain
SHA-bound provenance; they are never concatenated into a model's conversation.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict, deque
import hashlib
import json
import re
import shutil
from pathlib import Path
from threading import Lock

from .admission import (
    Admission, MAX_JSON_BYTES, SUPPORTED_DRAW_OBSERVATIONS,
    SUPPORTED_SYSTEM_SHA256, SUPPORTED_TOOLS_SHA256, _validate_assistant, _validate_draw_runtime_state,
    isolation_key, local_artifact, prompt_hash, read_json, read_json_snapshot,
    reservations, sha256_file, strict_json_loads,
)
from ..training.template import PROJECTION_VERSION, normalize_messages
from .io_helpers import ordered_bounded_map, validate_workers

CASE_FORMAT = "optics-gold-agent"
DATASET_FORMAT = "optics-gold-agent-sft"
KINDS = {"first_draw", "revision", "submit"}
REQUIRED_VERDICTS = {
    "first_draw": ("caption_preserves_constraints",),
    "revision": ("substantive_change", "addresses_primary_defect", "primary_defect_improved",
                 "hard_constraints_preserved", "no_new_major_defects"),
    "submit": ("hard_constraints_satisfied", "visible_requirements_satisfied", "inspectable",
               "no_material_defect"),
}
CAPTION_LABELS = ("Style", "Composition", "Materials and lighting", "Visible optical results", "Detail and focus")


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def action_sha256(message: dict) -> str:
    """Binding shared by the collector and loader; ignore transport-only state."""
    return canonical_sha256(normalize_messages([message])[0])


def _model_name(value):
    return value.get("model") if isinstance(value, dict) else value


def _review_provenance(case):
    """Same-model review is explicit provenance, never implicit independence.

    Keep historical distinct-author/judge admission unchanged. A teacher that
    also judges its own output requires the new producer's two explicit mode
    markers and an exact policy/author/judge model match. This does not weaken
    any of the downstream checks binding decisions to actual image evidence.
    """
    models = case.get("models", {})
    model_names = {field: _model_name(models.get(field)) for field in ("author", "judge", "generator")}
    if any(not isinstance(value, str) or not value.strip() for value in model_names.values()):
        raise ValueError("Gold model provenance requires named author, judge and generator")
    config = case.get("config", {})
    teacher_mode = config.get("policy_mode") == "teacher_api"
    relationship = config.get("judge_relationship")
    if teacher_mode or relationship in {"same_model_separate_context", "distinct_model_separate_context"}:
        if not teacher_mode or relationship not in {"same_model_separate_context", "distinct_model_separate_context"}:
            raise ValueError("Gold same-model review requires explicit teacher_api and same_model_separate_context markers")
        policy_model = _model_name(models.get("policy"))
        if relationship == "same_model_separate_context" and (
                policy_model != model_names["author"] or model_names["author"] != model_names["judge"]):
            raise ValueError("Gold same-model teacher review requires matching policy, author and judge models")
        if relationship == "distinct_model_separate_context" and (
                policy_model != model_names["author"] or model_names["author"] == model_names["judge"]):
            raise ValueError("Gold distinct-model teacher review requires policy matching author and excludes same-model judging")
        return model_names, {"review_type": "model", "reviewer": model_names["judge"],
                             "policy_mode": "teacher_api", "judge_relationship": relationship}
    if model_names["author"] == model_names["judge"]:
        raise ValueError("Gold model provenance needs distinct author and independent judge unless explicitly opted into same-model teacher review")
    return model_names, None


def _check_raw_evidence(evidence, kind):
    """Cross-check archived provider judgments against the collector's verdict.

    Initial schema-v1 producers may supply only the required hash-bound
    verdict fields. Once raw checks are archived, negative checks cannot be
    overwritten by a positive derived verdict in the same evidence record.
    """
    if "checks" not in evidence:
        return
    checks = evidence["checks"]
    if not isinstance(checks, dict):
        raise ValueError("Gold raw checks must be structured independent judge evidence")
    def positive(value, fields):
        if not isinstance(value, dict) or any(value.get(field) is not True for field in fields):
            raise ValueError("Gold derived verdict contradicts archived independent judge checks")
    def requirements(inspection):
        rows = inspection.get("requirements") if isinstance(inspection, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("Gold archived inspection must preserve per-requirement visual evidence")
        result = {}
        for row in rows:
            if (not isinstance(row, dict) or row.get("priority") not in {"hard", "derived", "inspectability"}
                    or row.get("status") not in {"satisfied", "contradicted", "not-verifiable"}
                    or not isinstance(row.get("requirement"), str) or not row["requirement"].strip()
                    or not isinstance(row.get("evidence"), str) or not row["evidence"].strip()):
                raise ValueError("Gold raw requirement evidence is invalid")
            key = (row["priority"], row["requirement"])
            if key in result:
                raise ValueError("Gold raw requirement evidence repeats a requirement")
            result[key] = row["status"]
        return result
    if kind == "first_draw":
        positive(checks.get("caption_gate"), ("preserves_hard_constraints", "standalone", "visible_results_concrete"))
        if "no_unseen_observation_claims" in checks["caption_gate"]:
            positive(checks["caption_gate"], ("no_unseen_observation_claims",))
        requirements(checks.get("inspection"))
    elif kind == "submit":
        inspection = checks.get("inspection")
        rows = requirements(inspection)
        positive(checks.get("submit_gate"), ("grounded", "correct_decision", "concise"))
        if (inspection.get("decision") != "submit" or inspection.get("material_defect") is not False
                or inspection.get("inspectable") is not True or any(state != "satisfied" for state in rows.values())):
            raise ValueError("Gold submit contradicts its actual image inspection")
    else:
        positive(checks.get("revision_gate"), ("addresses_primary_defect", "substantive_change",
                 "preserves_hard_constraints", "standalone", "diagnosis_supported"))
        comparison = checks.get("comparison")
        positive(comparison, ("primary_defect_improved",))
        if (comparison.get("hard_constraint_regression") is not False or comparison.get("new_material_defect") is not False
                or type(comparison.get("score_delta")) is not int or comparison["score_delta"] <= 0):
            raise ValueError("Gold revision contradicts its raw pairwise improvement evidence")
        before, after = requirements(checks.get("inspection_before")), requirements(checks.get("inspection_after"))
        if set(before) != set(after) or any(key[0] == "hard" and state == "satisfied" and after[key] != "satisfied"
                                           for key, state in before.items()):
            raise ValueError("Gold revision lost a previously satisfied hard constraint in raw inspection")


def _native(message):
    return {key: copy.deepcopy(value) for key, value in message.items() if key != "message_id"}


def _caption(message):
    return strict_json_loads(message["tool_calls"][0]["function"]["arguments"])["prompt"]


def _validate_caption(caption):
    found = list(re.finditer(r"(?m)^\s*(Style|Composition|Materials and lighting|Visible optical results|Detail and focus)\s*:\s*", caption))
    if tuple(match.group(1) for match in found) != CAPTION_LABELS or caption[:found[0].start()].strip():
        raise ValueError("Gold draw needs exactly the five caption sections in order")
    for index, match in enumerate(found):
        end = found[index + 1].start() if index + 1 < len(found) else len(caption)
        if not caption[match.end():end].strip():
            raise ValueError("Gold caption sections cannot be empty")


class _LoadDigestCache:
    """Reuse inventory digests only while files stay unchanged in this load.

    No bytes or parsed judgments are retained. Target evidence snapshots and
    the final source mutation check deliberately bypass this cache.
    """

    def __init__(self):
        self._digests = {}
        self._guard = Lock()
        # local_artifact resolves aliases before calling this cache. Striped
        # locks avoid hashing one resolved file concurrently without retaining
        # a lock for every entry in a large inventory.
        self._locks = [Lock() for _ in range(64)]

    @staticmethod
    def _state(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def __call__(self, path):
        with self._locks[hash(path) % len(self._locks)]:
            state = self._state(path)
            with self._guard:
                cached = self._digests.get(path)
            if cached is not None:
                digest, original_state = cached
                if state != original_state:
                    raise ValueError("Gold dataset file changed during admission")
                return digest
            digest = sha256_file(path)
            if self._state(path) != state:
                raise ValueError("Gold dataset file changed during admission")
            with self._guard:
                self._digests[path] = digest, state
            return digest


def _hash_inventory(case, root, file_map, *, evidence_paths=(), file_digest=None):
    assets, evidence, actual, snapshots = {}, {}, {}, {}
    for field, inventory in (("assets", assets), ("evidence_files", evidence)):
        entries = case.get(field)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Gold {field} needs a nonempty hash inventory")
        for entry in entries:
            if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))):
                raise ValueError("Gold file entries require path and lowercase SHA256")
            name = entry["path"]
            if Path(name).is_absolute() or ".." in Path(name).parts or name in actual:
                raise ValueError("Gold file paths must be unique relative paths")
            path = file_map.get(name) if file_map is not None else local_artifact(root, name)
            if path is None:
                raise ValueError("Gold evidence/asset SHA256 mismatch or missing file")
            if field == "evidence_files" and name in evidence_paths:
                # Validate the very bytes whose digest was checked. Retain only
                # this case's target judgments, avoiding a second remote read
                # and a hash/parse race; every admission starts fresh.
                snapshots[name], digest = read_json_snapshot(Path(path))
            else:
                digest = (file_digest or sha256_file)(Path(path))
            if digest != entry["sha256"]:
                raise ValueError("Gold evidence/asset SHA256 mismatch or missing file")
            inventory[name] = entry
            actual[name] = Path(path)
    return assets, evidence, actual, snapshots


def _replay(case, assets):
    """Match each actual draw to its exact tool result and observed pixels."""
    messages = case.get("messages")
    if not isinstance(messages, list) or len(messages) < 5:
        raise ValueError("Gold cases need system, request, an executed draw and its actual image")
    messages = [_native(message) for message in messages]
    if (set(messages[0]) != {"role", "content"} or messages[0]["role"] != "system"
            or not isinstance(messages[0]["content"], str)
            or hashlib.sha256(messages[0]["content"].encode()).hexdigest() not in SUPPORTED_SYSTEM_SHA256):
        raise ValueError("Gold system prompt must match the published Agent runtime")
    if messages[1] != {"role": "user", "content": case["task"]["prompt"]}:
        raise ValueError("Gold initial user message must be exactly the original request")
    tools = case.get("tools")
    if (not isinstance(tools, list) or len(tools) != 1
            or tools[0].get("function", {}).get("name") != "draw"
            or canonical_sha256(tools) not in SUPPORTED_TOOLS_SHA256):
        raise ValueError("Gold tools must use the published native draw-only interface")
    normalize_messages(messages)
    cursor, calls, draws, prior, prior_caption, decisions = 2, set(), 0, None, None, {}
    seen_pixels = set()
    while cursor < len(messages):
        action = messages[cursor]
        draw = _validate_assistant(action, tools)
        if not draw:
            if prior is None or cursor != len(messages) - 1 or len(action.get("content", "")) > 1000:
                raise ValueError("Gold submit must be brief, terminal and follow actual image observation")
            decisions[cursor] = {"kind": "submit", "image_path": prior, "previous_image_path": None}
            cursor += 1
            continue
        draws += 1
        if draws > 3 or cursor + 2 >= len(messages):
            raise ValueError("Gold chosen trajectory allows at most three fully executed draws")
        if len(action.get("content") or "") > 1600:
            raise ValueError("Gold public diagnosis must be short and paired with its action")
        caption = _caption(action)
        _validate_caption(caption)
        if prior_caption is not None and prompt_hash(caption) == prompt_hash(prior_caption):
            raise ValueError("Identical redraw cannot enter Gold SFT")
        call_id = action["tool_calls"][0]["id"]
        tool, observation = messages[cursor + 1:cursor + 3]
        if (call_id in calls or set(tool) != {"role", "name", "tool_call_id", "content"}
                or tool["role"] != "tool" or tool["name"] != "draw" or tool["tool_call_id"] != call_id
                or not isinstance(tool["content"], str)):
            raise ValueError("Gold draw needs its unique matching native tool result")
        calls.add(call_id)
        payload = strict_json_loads(tool["content"])
        if (not isinstance(payload, dict) or payload.get("status") != "ok"
                or set(payload) - {"status", "image_path", "sha256", "width", "height", "runtime_state"}):
            raise ValueError("Gold tool payload must describe successful generated pixels only")
        image = payload.get("image_path")
        if image not in assets or payload.get("sha256") != assets[image]["sha256"]:
            raise ValueError("Gold draw image must agree with its SHA-bound asset")
        for dimension in ("width", "height"):
            if dimension in payload and (type(payload[dimension]) is not int or payload[dimension] <= 0):
                raise ValueError("Gold draw dimensions must be positive integers")
            if dimension in assets[image] and payload.get(dimension) != assets[image][dimension]:
                raise ValueError("Gold draw dimensions disagree with its asset")
        if assets[image].get("kind", "generated_image") != "generated_image":
            raise ValueError("Gold draw observations must be generated images")
        if assets[image]["sha256"] in seen_pixels:
            raise ValueError("Byte-identical redraw images cannot be Gold corrections")
        seen_pixels.add(assets[image]["sha256"])
        if "runtime_state" in payload:
            _validate_draw_runtime_state(payload, case.get("config", {"max_image_calls": 3}),
                                         draw_number=draws, prompt=caption,
                                         image_sha256=assets[image]["sha256"], previous_prompt=prior_caption,
                                         previous_image_sha256=assets[prior]["sha256"] if prior else None)
        allowed_observations = [{"role": "user", "content": [
            {"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image}}
        ]} for text in SUPPORTED_DRAW_OBSERVATIONS]
        if observation not in allowed_observations:
            raise ValueError("Gold observation must contain precisely the actual returned image and runtime text")
        decisions[cursor] = {"kind": "first_draw" if prior is None else "revision",
                             "image_path": image, "previous_image_path": prior}
        prior, prior_caption = image, caption
        cursor += 3
    final = case.get("final", {})
    submitted = messages[-1]["role"] == "assistant"
    if (final.get("status") != ("submitted" if submitted else "verified_prefix")
            or final.get("image_path") != prior):
        raise ValueError("Gold final state must identify the observed latest image or verified prefix")
    return messages, decisions


def _admit_case(path, case, digest, *, file_map=None, file_digest=None):
    if case.get("schema_version") != 1 or case.get("format") != CASE_FORMAT:
        raise ValueError("Expected explicit optics-gold-agent schema_version 1")
    task = case.get("task", {})
    for field in ("task_id", "prompt", "category_id", "scene_family", "split"):
        if not isinstance(task.get(field), str) or not task[field].strip():
            raise ValueError(f"Gold task.{field} is required")
    if task["split"] not in {"train", "validation"}:
        raise ValueError("Benchmark/test cases never enter Gold SFT")
    models = case.get("models", {})
    model_names, review_provenance = _review_provenance(case)
    targets = case.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Gold cases require selected verified targets")
    evidence_paths = {target.get("evidence", {}).get("path") for target in targets}
    assets, evidence_files, actual, evidence_snapshots = _hash_inventory(
        case, path.parent, file_map, evidence_paths=evidence_paths, file_digest=file_digest)
    messages, decisions = _replay(case, assets)
    indexes, rows = set(), []
    for target in targets:
        index = target.get("message_index")
        if type(index) is not int or index not in decisions or index in indexes:
            raise ValueError("Gold targets must identify distinct actual assistant decisions")
        indexes.add(index)
        labels = target.get("labels")
        if (not isinstance(labels, list) or not labels or len(set(labels)) != len(labels)
                or any(label not in {"A", "B", "C", "D", "E", "F"} for label in labels)):
            raise ValueError("Gold labels must be nonempty unique A-F labels")
        info = decisions[index]
        if ((info["kind"] == "first_draw" and not set(labels) <= {"A", "B"})
                or (info["kind"] == "revision" and not set(labels) <= {"C", "D", "E"})
                or (info["kind"] == "submit" and set(labels) != {"E"})):
            raise ValueError("Gold A-F labels must match the actual selected action; F is provenance only")
        if "C" in labels and not (messages[index].get("content") or "").strip():
            raise ValueError("Gold visual-diagnosis label C requires actual public diagnosis in the target")
        evidence_path = target.get("evidence", {}).get("path")
        if evidence_path not in evidence_files:
            raise ValueError("Each target needs separately SHA-bound independent judge evidence")
        evidence = evidence_snapshots[evidence_path]
        expected = {"schema_version": 1, "kind": info["kind"], "message_index": index,
                    "task_prompt_sha256": prompt_hash(task["prompt"]),
                    "action_sha256": action_sha256(messages[index]),
                    "image_path": info["image_path"], "image_sha256": assets[info["image_path"]]["sha256"],
                    "judge_model": model_names["judge"]}
        if any(evidence.get(key) != value for key, value in expected.items()):
            raise ValueError("Gold judge evidence does not bind the actual action, request, image and judge")
        if info["kind"] == "revision":
            previous = info["previous_image_path"]
            if (evidence.get("previous_image_path") != previous
                    or evidence.get("previous_image_sha256") != assets[previous]["sha256"]):
                raise ValueError("Gold revision needs the actual preceding image comparison")
        verdict = evidence.get("verdict", {})
        required = ("passed", "actual_image_inspected", *REQUIRED_VERDICTS[info["kind"]])
        if any(verdict.get(field) is not True for field in required):
            raise ValueError("Gold judge rejected or did not verify a required action criterion")
        _check_raw_evidence(evidence, info["kind"])
        conversation, image_hashes = copy.deepcopy(messages[:index + 1]), []
        for message in conversation:
            for part in message.get("content", []) if isinstance(message.get("content"), list) else []:
                if part.get("type") == "image_url":
                    name = part["image_url"]["url"]
                    image_hashes.append(assets[name]["sha256"])
                    part["image_url"] = {"url": str(actual[name])}
        rows.append({"schema_version": 2, "sample_id": f"gold:{digest}:m{index}",
                     "trajectory_sha256": digest, "step_id": f"m{index}",
                     "scene_family": task["scene_family"], "task_id": task["task_id"], "category_id": task["category_id"],
                     "prompt_sha256": prompt_hash(task["prompt"]), "split": task["split"],
                     "original_split": task["split"], "is_demo": False,
                     "tools": copy.deepcopy(case["tools"]), "messages": conversation,
                     "target_message_index": index, "image_sha256": image_hashes,
                     "projection_version": PROJECTION_VERSION, "gold_labels": labels,
                     "gold_evidence_sha256": evidence_files[evidence_path]["sha256"]})
        if review_provenance is not None:
            rows[-1]["review"] = copy.deepcopy(review_provenance)
    if sha256_file(path) != digest:
        raise ValueError("Gold case changed during admission")
    metadata = {"trajectory_path": str(path), "trajectory_sha256": digest,
                "human_review_sha256": None, "machine_review_sha256": canonical_sha256(case["evidence_files"]),
                "selected_step_ids": [row["step_id"] for row in rows], "scene_family": task["scene_family"],
                "split": task["split"], "original_split": task["split"], "is_demo": False,
                "review_type": "model", "reviewer": model_names["judge"], "models": models,
                "artifact_paths": {name: str(path) for name, path in actual.items()},
                "artifact_sha256": {str(actual[name]): item["sha256"]
                                    for name, item in {**assets, **evidence_files}.items()}}
    if review_provenance is not None:
        metadata.update(review_provenance)
    return rows, metadata


def _decision_key(row):
    """Deduplicate the exact target/context across branches and transport IDs."""
    messages = normalize_messages(row["messages"])
    hashes = iter(row["image_sha256"])
    call_ids = {}
    for message in messages:
        for call in message.get("tool_calls", []):
            call_ids[call["id"]] = f"call_{len(call_ids)}"
            call["id"] = call_ids[call["id"]]
            call["function"]["arguments"] = strict_json_loads(call["function"]["arguments"])
        if message["role"] == "tool":
            message["tool_call_id"] = call_ids[message["tool_call_id"]]
            payload = strict_json_loads(message["content"])
            payload["image_path"] = "sha256:" + payload["sha256"]
            message["content"] = payload
        for part in message.get("content", []) if isinstance(message.get("content"), list) else []:
            if part.get("type") == "image":
                part["image"] = "sha256:" + next(hashes)
    return canonical_sha256({"messages": messages, "tools": row["tools"]})


def admit_gold_cases(paths, *, reserved_paths=None, artifact_maps=None, _file_digest=None,
                     validation_workers=1) -> Admission:
    """Keep at most one chosen trajectory per task; reject any cross-split group."""
    validate_workers(validation_workers)
    loaded, rejected, groups = [], [], {key: {} for key in ("task_ids", "scene_families", "prompt_sha256", "image_sha256")}
    reserved = reservations(reserved_paths)

    def validate_case(path):
        try:
            case, digest = read_json_snapshot(path)
            rows, metadata = _admit_case(path, case, digest, file_map=(artifact_maps or {}).get(str(path)),
                                        file_digest=_file_digest)
            task = case["task"]
            identity = {"task_ids": {isolation_key(task["task_id"])},
                        "scene_families": {isolation_key(task["scene_family"])},
                        "prompt_sha256": {prompt_hash(task["prompt"])},
                        "image_sha256": {item["sha256"] for item in case["assets"]}}
            return (rows, metadata, identity), task["split"], None
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
            return None, None, {"trajectory_path": str(path), "reason": str(error)}

    ordered_paths = sorted({Path(path).resolve() for path in paths})
    for entry, split, rejection in ordered_bounded_map(validate_case, ordered_paths, workers=validation_workers):
        if rejection is not None:
            rejected.append(rejection)
            continue
        for key, values in entry[2].items():
            for value in values:
                groups[key].setdefault(value, set()).add(split)
        loaded.append(entry)
    samples, accepted, seen_tasks, seen_targets = [], [], set(), set()
    # Identical ordering before and after relocation/content-addressed export.
    loaded.sort(key=lambda entry: (next(iter(entry[2]["task_ids"])), entry[1]["trajectory_sha256"]))
    for rows, metadata, identity in loaded:
        reason = None
        for key, values in identity.items():
            if values & reserved[key]:
                reason = f"Reserved benchmark/test overlap: {key}"
            if any(len(groups[key][value]) > 1 for value in values):
                reason = f"Cross-split leakage: {key}"
        task_id = next(iter(identity["task_ids"]))
        if task_id in seen_tasks:
            reason = "At most one Gold trajectory per original task"
        if reason:
            rejected.append({"trajectory_path": metadata["trajectory_path"], "reason": reason})
            continue
        retained = []
        for row in rows:
            key = _decision_key(row)
            if key not in seen_targets:
                seen_targets.add(key)
                retained.append(row)
        if retained:
            metadata["selected_step_ids"] = [row["step_id"] for row in retained]
            samples.extend(retained)
            accepted.append(metadata)
            seen_tasks.add(task_id)
        else:
            rejected.append({"trajectory_path": metadata["trajectory_path"], "reason": "All selected targets duplicate existing causal examples"})
    return Admission(samples, accepted, rejected)


def _portable_row(row, path_map):
    result = copy.deepcopy(row)
    images = []
    for message in result["messages"]:
        for part in message.get("content", []) if isinstance(message.get("content"), list) else []:
            if part.get("type") == "image_url":
                part["image_url"]["url"] = path_map[part["image_url"]["url"]]
                images.append(part["image_url"]["url"])
    result["images"] = images
    return result


def _target_kind(row):
    if not row["messages"][-1].get("tool_calls"):
        return "submit"
    return "first_draw" if row["target_message_index"] == 2 else "revision"


def _select_targets(admission, target_limits):
    """Bound common first/submit actions while preserving every useful revision."""
    if target_limits is None:
        return admission.samples, {row["trajectory_sha256"]: list(row["selected_step_ids"]) for row in admission.accepted}
    if (not isinstance(target_limits, dict) or set(target_limits) - {"first_draw", "submit"}
            or any(type(value) is not int or value < 0 for value in target_limits.values())):
        raise ValueError("Gold target_limits only supports nonnegative first_draw and submit caps; revisions are never capped")
    priorities = {}
    revision_sources = {row["trajectory_sha256"] for row in admission.samples if "D" in row["gold_labels"]}
    for accepted in admission.accepted:
        flags = set()
        for original, actual in accepted["artifact_paths"].items():
            if Path(original).name == "rejected_actions.json":
                records = read_json(Path(actual))
                if not isinstance(records, list):
                    raise ValueError("Gold rejected_actions provenance must be a list")
                flags.update(record.get("failure_type") for record in records if isinstance(record, dict))
        priority = (0 if flags & {"no_draw", "identical_redraw", "max_try_reached"} else
                    1 if flags & {"near_identical_redraw", "premature_submit", "wrong_diagnosis", "lost_user_constraint",
                                  "unnecessary_redraw", "invalid_tool_call", "ungrounded_submit"} else
                    2 if accepted["trajectory_sha256"] in revision_sources else 3)
        priorities[accepted["trajectory_sha256"]] = priority
    by_priority = defaultdict(lambda: defaultdict(deque))
    for row in sorted(admission.samples, key=lambda row: (isolation_key(row["task_id"]), row["target_message_index"], row["sample_id"])):
        by_priority[priorities[row["trajectory_sha256"]]][row["category_id"]].append(row)
    ordered = []
    for priority in sorted(by_priority):
        categories = by_priority[priority]
        while categories:
            for category in sorted(list(categories)):
                ordered.append(categories[category].popleft())
                if not categories[category]:
                    del categories[category]
    chosen, counts = set(), {"first_draw": 0, "submit": 0}
    for row in ordered:
        kind = _target_kind(row)
        if kind in target_limits and counts[kind] >= target_limits[kind]:
            continue
        chosen.add(row["sample_id"])
        if kind in counts:
            counts[kind] += 1
    rows = [row for row in admission.samples if row["sample_id"] in chosen]
    selection = {source["trajectory_sha256"]: [] for source in admission.accepted}
    for row in rows:
        selection[row["trajectory_sha256"]].append(row["step_id"])
    return rows, selection


def export_gold_dataset(paths, output_dir, *, reserved_paths=None, target_limits=None,
                        artifact_maps=None, validation_workers=1):
    """Export in deterministic order with bounded validation/copy concurrency."""
    admission = admit_gold_cases(paths, reserved_paths=reserved_paths,
                                artifact_maps=artifact_maps, validation_workers=validation_workers)
    if not admission.samples:
        raise ValueError("No verified Gold decision samples: " + json.dumps(admission.rejected, ensure_ascii=False))
    selected, selection = _select_targets(admission, target_limits)
    if not selected:
        raise ValueError("Gold target limits selected no verified assistant actions")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Gold dataset output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    files, sources, path_map, source_counts = {}, [], {}, Counter()
    # Multiple trajectories can share a content-addressed image destination.
    # Serialize each destination's existence check, copy and digest checks;
    # keep this bounded without one lock per artifact. Manifest mutations stay
    # on the consuming thread, in exactly the same order as the serial export.
    copy_locks = [Lock() for _ in range(64)]

    def copy_file(source, relative, expected):
        target = output / relative
        with copy_locks[hash(relative) % len(copy_locks)]:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            # Recheck every source, even when another trajectory has already
            # supplied the same destination. Shared pixels do not authorize
            # changed bytes in a different original source file.
            if sha256_file(target) != expected or sha256_file(Path(source)) != expected:
                raise ValueError("Gold source changed during portable export")

    def copy_source(accepted):
        source_files, source_images = [], {}
        path, digest = Path(accepted["trajectory_path"]), accepted["trajectory_sha256"]
        relative = f"provenance/{digest}/gold.json"
        copy_file(path, relative, digest)
        source_files.append((relative, digest))
        case, actual_digest = read_json_snapshot(path)
        if actual_digest != digest:
            raise ValueError("Gold source changed during portable export")
        source = {"gold": relative, "sha256": digest, "artifact_map": {}, "selected_step_ids": selection[digest]}
        assets = {item["path"] for item in case["assets"]}
        for original, actual in accepted["artifact_paths"].items():
            sha = accepted["artifact_sha256"][actual]
            suffix = Path(original).suffix.lower()
            target = (f"assets/{sha}{suffix}" if original in assets else f"provenance/{digest}/evidence/{sha}{suffix}")
            copy_file(actual, target, sha)
            source_files.append((target, sha))
            source["artifact_map"][original] = target
            if original in assets:
                source_images[actual] = target
        return source, source_files, source_images, case["final"]["status"]

    for source, source_files, source_images, status in ordered_bounded_map(
            copy_source, admission.accepted, workers=validation_workers):
        files.update(source_files)
        path_map.update(source_images)
        source_counts[status] += 1
        sources.append(source)
    for split in ("train", "validation", "demo"):
        target = output / f"{split}.jsonl"
        with target.open("w", encoding="utf-8") as stream:
            for row in selected:
                if row["split"] == split:
                    stream.write(json.dumps(_portable_row(row, path_map), ensure_ascii=False, allow_nan=False) + "\n")
        files[target.name] = sha256_file(target)
    embedded = []
    for path in reserved_paths or []:
        digest = sha256_file(Path(path))
        relative = f"reservations/{digest}.json"
        copy_file(path, relative, digest)
        files[relative] = digest
        if relative not in embedded:
            embedded.append(relative)
    manifest = {"schema_version": 1, "format": DATASET_FORMAT, "portable": True,
                "projection_version": PROJECTION_VERSION, "files": files, "sources": sources,
                "reserved_manifests": embedded, "target_limits": target_limits,
                "selection_policy": "failure-recovery-priority-v1",
                "raw_counts": {split: sum(row["split"] == split for row in admission.samples)
                               for split in ("train", "validation", "demo")},
                "counts": {split: sum(row["split"] == split for row in selected)
                                                            for split in ("train", "validation", "demo")},
                "label_counts": dict(Counter(label for row in selected for label in row["gold_labels"])),
                "target_kind_counts": dict(Counter(_target_kind(row) for row in selected)),
                "source_counts": {**dict(source_counts), "total": len(sources),
                                  "with_selected_targets": sum(bool(source["selected_step_ids"]) for source in sources)},
                "rejected": admission.rejected,
                "loss_scope": "one selected assistant text/native tool call; all context, tool and image tokens masked",
                "source_scope": "Full Gold messages/evidence/rejected actions are provenance, never appended to inputs"}
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def load_gold_dataset(dataset_dir, *, reserved_paths=None, validation_workers=1):
    validate_workers(validation_workers)
    root = Path(dataset_dir).expanduser().resolve()
    manifest = read_json(root / "dataset_manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("format") != DATASET_FORMAT
            or manifest.get("portable") is not True or manifest.get("projection_version") != PROJECTION_VERSION):
        raise ValueError("Unsupported portable Gold dataset schema/projection")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Gold dataset requires a hash inventory")
    file_digest = _LoadDigestCache()
    def validate_file(item):
        relative, digest = item
        if Path(relative).is_absolute() or file_digest(local_artifact(root, relative)) != digest:
            raise ValueError("Gold dataset file SHA256 mismatch")

    for _ in ordered_bounded_map(validate_file, files.items(), workers=validation_workers):
        pass

    def resolve_source(source):
        if source.get("gold") not in files or source.get("sha256") != files[source["gold"]]:
            raise ValueError("Gold source missing from hash inventory")
        path = local_artifact(root, source["gold"])
        artifacts = {}
        relative_paths = {}
        for original, relative in source["artifact_map"].items():
            if relative not in files:
                raise ValueError("Gold artifact missing from hash inventory")
            actual = local_artifact(root, relative)
            artifacts[original] = actual
            relative_paths[str(actual)] = relative
        return path, artifacts, relative_paths

    paths, maps, path_map = [], {}, {}
    seen_paths = set()
    for path, artifacts, relative_paths in ordered_bounded_map(
            resolve_source, manifest.get("sources", []), workers=validation_workers):
        if path in seen_paths:
            raise ValueError("Duplicate Gold source")
        seen_paths.add(path)
        paths.append(path)
        maps[str(path)] = artifacts
        path_map.update(relative_paths)
    embedded = manifest.get("reserved_manifests", [])
    if not isinstance(embedded, list) or any(name not in files for name in embedded):
        raise ValueError("Gold reservations missing from hash inventory")
    reservation_paths = [*(reserved_paths or []), *(local_artifact(root, name) for name in embedded)]
    # Preserve selection order: content-addressed provenance ordering can differ
    # from source directories when several tasks share a duplicate first action.
    admitted = admit_gold_cases(paths, reserved_paths=reservation_paths, artifact_maps=maps,
                               _file_digest=file_digest, validation_workers=validation_workers)
    if admitted.rejected:
        raise ValueError("Portable Gold no longer matches verified raw sources: " + json.dumps(admitted.rejected))
    if manifest.get("selection_policy", "failure-recovery-priority-v1") != "failure-recovery-priority-v1":
        raise ValueError("Unsupported Gold target-selection policy")
    selected, selection = _select_targets(admitted, manifest.get("target_limits"))
    for field, expected_counts in (
        ("label_counts", dict(Counter(label for row in selected for label in row["gold_labels"]))),
        ("target_kind_counts", dict(Counter(_target_kind(row) for row in selected))),
    ):
        if field in manifest and manifest[field] != expected_counts:
            raise ValueError("Gold selected label/kind counts do not match verified targets")
    for source in manifest["sources"]:
        if source.get("selected_step_ids", selection[source["sha256"]]) != selection[source["sha256"]]:
            raise ValueError("Gold selected_step_ids contradict the declared target-selection policy")
    admitted.samples = selected
    admitted.accepted = [source for source in admitted.accepted if selection[source["trajectory_sha256"]]]
    for source in admitted.accepted:
        source["selected_step_ids"] = selection[source["trajectory_sha256"]]
    expected = {row["sample_id"]: _portable_row(row, path_map) for row in admitted.samples}
    ordered, seen = [], set()
    for split in ("train", "validation", "demo"):
        if f"{split}.jsonl" not in files:
            raise ValueError("Gold decision JSONL missing from hash inventory")
        count = 0
        with local_artifact(root, f"{split}.jsonl").open("rb") as stream:
            while line := stream.readline(MAX_JSON_BYTES + 1):
                if len(line) > MAX_JSON_BYTES:
                    raise ValueError("Gold decision exceeds bounded JSON size")
                row = strict_json_loads(line.decode("utf-8"))
                sample_id = row.get("sample_id")
                if row.get("split") != split or sample_id in seen or expected.get(sample_id) != row:
                    raise ValueError("Gold selected row no longer matches verified source action")
                seen.add(sample_id)
                ordered.append(sample_id)
                count += 1
        if count != manifest.get("counts", {}).get(split):
            raise ValueError("Gold split count mismatch")
    if seen != set(expected):
        raise ValueError("Gold selected actions missing from portable rows")
    rows = {row["sample_id"]: row for row in admitted.samples}
    admitted.samples = [rows[key] for key in ordered]
    return admitted
