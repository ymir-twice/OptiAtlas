"""Machine review is explicit and distinct from the legacy human-review path."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from optics_agent_sft.curation.admission import (
    admit_trajectories, local_artifact, read_json_snapshot, sha256_file, trajectory_paths,
)

from .logging import atomic_json, fingerprint, utcnow
from .rewards import make_reward
from optics_agent.protocol import LEGACY_SYSTEM_PROMPT, V2_SYSTEM_PROMPT, SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT
from optics_agent.protocol_v3 import V3_SYSTEM_PROMPT
from optics_agent.protocol_react_v1 import REACT_V1_SYSTEM_PROMPT
from optics_agent.protocol_react_v2 import REACT_V2_SYSTEM_PROMPT
from .protocols import LEGACY_UMM_SYSTEM, V2_UMM_SYSTEM, UMM_SYSTEM, RESEARCH_UMM_SYSTEM
from .protocols_v3 import V3_UMM_SYSTEM
from .protocols_react_v1 import REACT_V1_UMM_SYSTEM
from .protocols_react_v2 import REACT_V2_UMM_SYSTEM


def review_runs(cfg, logger=None, *, reward=None):
    reward = make_reward(cfg["reward"]) if reward is None else reward
    judge_config = {k: v for k, v in cfg["reward"].items()
                    if k not in {"max_requests", "timeout", "api_key_env", "allow_no_key", "max_image_bytes"}}
    review_identity = {"reward_config_sha256": fingerprint(judge_config),
                       "judge": getattr(reward, "identity", None),
                       "min_optical_score": float(cfg.get("min_optical_score", .8)),
                       "min_prompt_score": float(cfg.get("min_prompt_score", .8)),
                       "selection_policy": "all_completed_steps_of_final_accepted_trajectory"}
    reviewed, skipped, failed = [], [], []
    max_items = int(cfg.get("max_items", 100))
    if max_items < 1:
        raise ValueError("agent_data.max_items must be positive")
    for path in trajectory_paths(cfg["runs"]):
        target = path.with_name("machine_review.json")
        obj, digest = read_json_snapshot(path)
        if obj.get("is_demo") or obj.get("task", {}).get("split") not in {"train", "validation"}:
            failed.append({"trajectory": str(path), "reason": "not_training_trajectory"})
            continue
        if target.exists():
            prior, _ = read_json_snapshot(target)
            if prior.get("trajectory_sha256") != digest:
                raise ValueError("Existing machine review belongs to different trajectory bytes")
            if prior.get("review_identity") != review_identity:
                raise ValueError("Existing machine review uses a different judge or thresholds; preserve it and use a separate review archive")
            final = obj.get("final", {})
            if sha256_file(local_artifact(path.parent, final["image_path"])) != final.get("image_sha256"):
                raise ValueError("Reviewed final image changed before resume")
            skipped.append(str(path))
            continue
        if len(reviewed) >= max_items:
            break
        final = obj.get("final", {})
        if final.get("status") not in {"submitted", "success"}:
            failed.append({"trajectory": str(path), "reason": "no_submission"})
            continue
        image = local_artifact(path.parent, final["image_path"])
        if sha256_file(image) != final.get("image_sha256"):
            raise ValueError("Final image changed before machine review")
        judgement = reward.score(obj["task"]["prompt"], image)
        approved = (judgement.get("assessable") is True and
                    judgement.get("optical_score", -1) >= float(cfg.get("min_optical_score", 0.8)) and
                    judgement.get("prompt_score", -1) >= float(cfg.get("min_prompt_score", 0.8)))
        judge = judgement.get("judge", {})
        if not judge.get("model") or not judge.get("prompt_sha256"):
            raise ValueError("Machine curation requires a judge model and prompt identity")
        steps = [step["step_id"] for step in obj["steps"]
                 if step.get("result", {}).get("status") in {"ok", "submitted"}]
        if not steps:
            approved = False
        review = {"schema_version": 2, "review_type": "model", "reviewer": judge["model"],
                  "reviewed_at": utcnow(), "decision": "approved" if approved else "rejected",
                  "trajectory_sha256": digest, "selected_step_ids": steps if approved else [],
                  "review_identity": review_identity,
                  "final_approved": approved, "judge": judge, "judgement": judgement,
                  "selection_policy": "all_completed_steps_of_final_accepted_trajectory",
                  "thresholds": {"optical": cfg.get("min_optical_score", 0.8),
                                 "prompt": cfg.get("min_prompt_score", 0.8)},
                  "final_review": {"optical_validity": "Machine assessment: " + judgement.get("evidence", ""),
                                   "prompt_fulfillment": "Machine assessment: " + judgement.get("evidence", "")},
                  "note": "Automatic curation; not a human annotation or proof of physical correctness."}
        if sha256_file(path) != digest or sha256_file(image) != final["image_sha256"]:
            raise ValueError("Trajectory/image changed during review")
        # Exclusive creation prevents concurrent reviewers from replacing evidence.
        with target.open("x", encoding="utf-8") as stream:
            json.dump(review, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        reviewed.append({"trajectory": str(path), "approved": approved})
        if logger:
            logger.log({"curation/reviewed": len(reviewed), "curation/approved": sum(x["approved"] for x in reviewed)})
    return {"reviewed": reviewed, "skipped": skipped, "failed": failed}


def _sense_messages(messages):
    """Keep observed images in context, and serialize native actions explicitly."""
    result = []
    for source in messages:
        role = source["role"]
        content = source.get("content") or ""
        if role == "system":
            # Preserve the instruction/target relation of historical trajectories.
            # A new contract must not retroactively relabel old draw captions.
            if content == RESEARCH_SYSTEM_PROMPT:
                projected_system = RESEARCH_UMM_SYSTEM
            elif content == SYSTEM_PROMPT:
                projected_system = UMM_SYSTEM
            elif content == REACT_V2_SYSTEM_PROMPT:
                projected_system = REACT_V2_UMM_SYSTEM
            elif content == REACT_V1_SYSTEM_PROMPT:
                projected_system = REACT_V1_UMM_SYSTEM
            elif content == V3_SYSTEM_PROMPT:
                projected_system = V3_UMM_SYSTEM
            elif content == V2_SYSTEM_PROMPT:
                projected_system = V2_UMM_SYSTEM
            elif content == LEGACY_SYSTEM_PROMPT:
                projected_system = LEGACY_UMM_SYSTEM
            else:
                raise ValueError("Cannot project an unrecognized Agent system protocol")
            result.append({"role": "system", "content": projected_system})
            continue
        if isinstance(content, list):
            content = [{"type": "text", "text": part["text"]} if part["type"] == "text" else
                       {"type": "image", "path": part["image_url"]["url"]} for part in content]
        if source.get("tool_calls"):
            action = source["tool_calls"][0]["function"]
            content = json.dumps({"action": action["name"], **json.loads(action["arguments"]),
                                  "observation": content if isinstance(content, str) else ""}, ensure_ascii=False)
        elif role == "assistant":
            content = json.dumps({"action": "submit", "text": content}, ensure_ascii=False)
        if role == "tool":
            role = "user"
            content = "Tool result: " + str(content)
        result.append({"role": role, "content": content})
    return result


def sense_rows_from_admission(admission):
    """Derive exact Sense supervision from freshly admitted immutable sources.

    Side-effect free: both export and training admission call this projection.
    """
    # Admission records the exact reference pixels, intermediate draws and tool
    # snapshots. Recheck them before projecting contexts, not just the target.
    for meta in admission.accepted:
        for artifact, expected_hash in meta["artifact_sha256"].items():
            if sha256_file(Path(artifact)) != expected_hash:
                raise ValueError("Admitted artifact changed during trajectory export")
    rows = admission.samples
    sense_rows = []
    for row in rows:
        messages = _sense_messages(row["messages"])
        sense_rows.append({"id": row["sample_id"], "kind": "text", "messages": messages[:-1],
                           "target_text": messages[-1]["content"], "split": row["split"],
                           "scene_family": row["scene_family"], "trajectory_sha256": row["trajectory_sha256"]})
    # Image supervision uses exactly the draw's causal context and its real output.
    selected = {row["sample_id"] for row in rows}
    for meta in admission.accepted:
        path = Path(meta["trajectory_path"])
        obj, digest = read_json_snapshot(path)
        if digest != meta["trajectory_sha256"]:
            raise ValueError("Trajectory changed during image target export")
        by_step = {row["step_id"]: row for row in rows if row["trajectory_sha256"] == digest}
        for step in obj["steps"]:
            key = f"{obj['trajectory_id']}:{step['step_id']}"
            if key not in selected:
                continue
            decision = by_step[step["step_id"]]
            calls = decision["messages"][-1].get("tool_calls", [])
            if not calls or calls[0]["function"]["name"] != "draw":
                continue
            result = step.get("result", {})
            # Runtime places artifact paths in result; resolve known generated artifacts
            # through the native tool response rather than fabricating a final-image pair.
            response_id = calls[0]["id"]
            responses = [m for m in obj.get("messages", []) if m.get("role") == "tool" and m.get("tool_call_id") == response_id]
            if len(responses) != 1:
                raise ValueError("Image export requires exactly one corresponding tool response")
            tool_result = json.loads(responses[0]["content"])
            image_key = tool_result.get("image_path") or result.get("image_path")
            image = local_artifact(path.parent, image_key)
            # A final-only judge does not approve earlier failed generations.
            # Keep those pixels as causal observations for correction decisions,
            # but supervise image synthesis only on the reviewed final output.
            if image_key != obj["final"]["image_path"]:
                continue
            if sha256_file(image) != obj["final"]["image_sha256"]:
                raise ValueError("Reviewed final-image hash changed during export")
            caption = json.loads(calls[0]["function"]["arguments"])["prompt"]
            sense_rows.append({"id": key + ":image", "kind": "image",
                "messages": _sense_messages(decision["messages"]), "caption": caption,
                "target_image": str(image), "image": str(image), "image_sha256": sha256_file(image),
                "split": decision["split"], "scene_family": decision["scene_family"],
                "trajectory_sha256": digest})
    return sense_rows


def prepare(cfg, logger=None):
    policy = tuple(cfg.get("accepted_review_types", ["human", "model"]))
    admission = admit_trajectories(cfg["runs"], reserved_paths=cfg.get("reserved_manifests", []),
                                  accepted_review_types=policy)
    output = Path(cfg["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not admission.samples:
        atomic_json(output / "admission.json", admission.summary())
        raise ValueError("No admitted Agent training samples; inspect admission.json")
    rows = admission.samples
    sense_rows = sense_rows_from_admission(admission)
    paths = {}
    for name, values in (("qwen_decisions", rows), ("sensenova_decisions", sense_rows)):
        for split in ("train", "validation"):
            dest = output / f"{name}_{split}.jsonl"
            if dest.exists():
                raise FileExistsError(f"Refusing to overwrite prepared data: {dest}")
            with dest.open("x", encoding="utf-8") as stream:
                for value in values:
                    if value["split"] == split:
                        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            paths[dest.name] = {"path": str(dest), "sha256": sha256_file(dest),
                                "rows": sum(x["split"] == split for x in values)}
    manifest = {"schema_version": 1, "review_policy": list(policy), "files": paths,
                "admission": admission.summary(), "created_at": utcnow(),
                "image_target_policy": "reviewed_final_draw_only",
                "reserved_manifests": [str(Path(p).resolve()) for p in cfg.get("reserved_manifests", [])],
                "reserved_manifest_sha256": {str(Path(p).resolve()): sha256_file(Path(p)) for p in cfg.get("reserved_manifests", [])},
                "portable": False, "note": "Image paths point to immutable original trajectories; preserve them."}
    atomic_json(output / "dataset_manifest.json", manifest)
    return manifest


def run(cfg, logger=None):
    action = cfg.get("action", "prepare")
    if action == "review":
        return review_runs(cfg, logger)
    if action == "prepare":
        return prepare(cfg, logger)
    if action == "validate":
        return admit_trajectories(cfg["runs"], reserved_paths=cfg.get("reserved_manifests", []),
              accepted_review_types=tuple(cfg.get("accepted_review_types", ["human", "model"]))).summary()
    raise ValueError("agent_data.action must be review, prepare or validate")
