"""One attempt's lifecycle, budgets and schema-2 archive around Qwen-Agent.

Read run() first: validate/create the attempt, consume OpticalAgent's native
message loop, optionally evaluate only after submission, then finalize the record.
The remaining methods are recorder callbacks invoked by agent.py and draw.

Model/tool messages are archived in causal order. Images are recorded with hashes
before being returned as observations. Counters are charged before calls, so a
failed request still consumes its call budget. Failures keep a terminal archive;
no evaluator or human-review feedback is ever inserted into the actor context.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .messages import PROJECTION_VERSION, project_messages
from .protocol import RESEARCH_PROTOCOL_VERSION, validate_evaluation, validate_task
from .providers import ImageProvider, ProviderError, safe_media_path
from .runtime_state import draw_runtime_state

if TYPE_CHECKING:
    from .llm import NativeChatModel

FinalEvaluator = Callable[[dict[str, Any], str, Path, float], dict[str, Any]]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Flush a complete JSON snapshot before replacing the previous archive."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_research_config(value: dict[str, Any] | None) -> dict[str, Any]:
    """Accept public settings only; secrets are read by providers from named envs."""
    value = {"enabled": False} if value is None else copy.deepcopy(value)
    if not isinstance(value, dict) or set(value) - {"enabled", "knowledge", "image_search"}:
        raise ValueError("research_tools supports enabled, knowledge and image_search only")
    if type(value.get("enabled", False)) is not bool:
        raise ValueError("research_tools.enabled must be boolean")
    if value.get("knowledge", {}) != {}:
        raise ValueError("research_tools.knowledge currently accepts only an empty object")
    search = value.get("image_search", {})
    if not isinstance(search, dict) or set(search) - {
            "provider", "api_key_env", "max_results", "max_image_side", "max_image_bytes"}:
        raise ValueError("Unsupported reference search settings; credentials must use api_key_env")
    if search.get("provider", "pexels") != "pexels":
        raise ValueError("Reference search supports pexels")
    if "api_key_env" in search and (not isinstance(search["api_key_env"], str) or not search["api_key_env"].isidentifier()):
        raise ValueError("api_key_env must name an environment variable")
    if "max_results" in search and (type(search["max_results"]) is not int or not 1 <= search["max_results"] <= 2):
        raise ValueError("Reference search max_results must be between 1 and 2")
    for name in ("max_image_side", "max_image_bytes"):
        if name in search and (type(search[name]) is not int or search[name] < 1):
            raise ValueError(name + " must be a positive integer")
    return value


@dataclass
class RunConfig:
    """Per-attempt limits; timeouts use the smaller of request and run time left."""
    max_steps: int = 8
    max_image_calls: int = 3
    max_knowledge_calls: int = 3
    max_search_calls: int = 3
    max_provider_calls: int = 16
    max_reported_tokens: int = 100000
    max_context_characters: int = 180000
    max_seconds: float = 900
    request_timeout: float = 180
    live_enabled: bool = False
    expose_runtime_state: bool = False

    def __post_init__(self):
        if not isinstance(self.expose_runtime_state, bool):
            raise ValueError("expose_runtime_state must be boolean")
        for name in ("max_steps", "max_image_calls", "max_provider_calls", "max_reported_tokens", "max_context_characters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        for name in ("max_knowledge_calls", "max_search_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        for name in ("max_seconds", "request_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")


class BudgetExceeded(RuntimeError):
    """A hard limit reached before the next action can be accepted."""
    pass


class AgentRunner:
    """Single-use coordinator; create a fresh instance for each independent task.

    OpticalAgent owns decision flow. This recorder owns persistent state and
    enforces budgets at the boundaries where the actor or draw provider is used.
    Providers may be replaced with explicit offline fixtures in tests, never by
    silently falling back to demo mode after a live failure.
    """
    def __init__(self, actor: NativeChatModel, image_provider: ImageProvider,
                 config: RunConfig | None = None, final_evaluator: FinalEvaluator | None = None,
                 *, research_tools: dict[str, Any] | None = None,
                 knowledge_lookup_callback=None, reference_search=None) -> None:
        self.actor, self.image_provider = actor, image_provider
        self.config = config or RunConfig()
        self.final_evaluator = final_evaluator
        self.research_tools = validate_research_config(research_tools)
        self.research_enabled = self.research_tools.get("enabled", False)
        self.knowledge_lookup_callback, self.reference_search = knowledge_lookup_callback, reference_search
        if self.research_enabled:
            if self.knowledge_lookup_callback is None:
                from .knowledge import knowledge_lookup
                self.knowledge_lookup_callback = knowledge_lookup
            if self.reference_search is None:
                from .tools.image_search import ReferenceImageSearch
                self.reference_search = ReferenceImageSearch(self.research_tools.get("image_search", {}))
        self._used = False

    def run(self, task: dict[str, Any], out_dir: str | Path, is_demo: bool = False,
            *, debug: bool = False) -> dict[str, Any]:
        """Execute, archive, and return one attempt; never overwrite an existing run.

        Before a directory is created, invalid task/mode/config errors propagate.
        After execution starts, caught errors become safe terminal records: only
        provider codes/request IDs or exception class names are written to disk.
        debug=True re-raises the caught exception after final archive persistence;
        it does not enable live calls or change success/review status. Ctrl-C is
        always re-raised after persistence, ensuring collection stops as well.
        """
        from .agent import OpticalAgent
        validate_task(task)
        if not isinstance(debug, bool):
            raise ValueError("debug must be boolean")
        if self._used:
            raise ValueError("Use a new runner for each independent attempt")
        providers = [self.actor, self.image_provider]
        if self.research_enabled:
            providers.append(self.reference_search)
        flags = [bool(getattr(provider, "is_demo", False)) for provider in providers]
        if is_demo and not all(flags):
            raise ValueError("Demo requires explicitly marked offline providers")
        if not is_demo and (any(flags) or not self.config.live_enabled):
            raise ValueError("Live execution requires real providers and explicit live_enabled")
        self.root = Path(out_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "images").mkdir()
        self._used = True
        self.started, self.event_number = time.monotonic(), 0
        self.deadline = self.started + self.config.max_seconds
        self.current_image, self.current_step, self.current_call = None, None, None
        self._last_draw_prompt: str | None = None
        self.counts = {"provider_calls": 0, "actor_calls": 0, "image_calls": 0,
                       "reported_total_tokens": 0, "calls_without_token_usage": 0}
        self.trajectory = {
            "schema_version": 2, "trajectory_id": "optics_" + uuid.uuid4().hex,
            "is_demo": bool(is_demo), "task": copy.deepcopy(task),
            "framework": {"name": "qwen-agent", "version": importlib.metadata.version("qwen-agent"),
                          "message_projection": PROJECTION_VERSION},
            "models": {"actor": self.actor.identity, "image": self.image_provider.identity},
            "tools": [], "messages": [], "steps": [], "artifacts": [],
            "started_at": utcnow(), "config": asdict(self.config),
            "final": {"status": "running", "image_path": None,
                      "evaluation": {"status": "not_run"}, "human_review_status": "pending"}}
        if self.research_enabled:
            (self.root / "research").mkdir()
            self.counts.update(knowledge_calls=0, search_calls=0)
            self.trajectory["research_tools"] = copy.deepcopy(self.research_tools)
            self.trajectory["framework"]["agent_protocol"] = RESEARCH_PROTOCOL_VERSION
            self.trajectory["models"]["reference_search"] = copy.deepcopy(self.reference_search.identity)
        self.event("run_started", {"trajectory_id": self.trajectory["trajectory_id"], "is_demo": is_demo})
        self.save()
        failure_to_raise: BaseException | None = None
        try:
            agent = OpticalAgent(self.actor, self)
            # Exactly the original prompt: no task id, scene family, questions,
            # rubric, reference image or hidden metadata is exposed to the actor.
            for _ in agent.run([{"role": "user", "content": task["prompt"]}]):
                pass
            if self.trajectory["final"]["status"] == "submitted" and self.final_evaluator is not None:
                self._external_evaluation(task)
        except BudgetExceeded as exc:
            self.fail("budget_exhausted", str(exc))
            if debug:
                failure_to_raise = exc
        except ProviderError as exc:
            details = {"request_id": exc.request_id}
            if exc.error_type:
                details["error_type"] = exc.error_type
            self.fail("technical_failure", exc.code, **details)
            if debug:
                failure_to_raise = exc
        except KeyboardInterrupt as exc:
            self.fail("interrupted", "keyboard_interrupt")
            failure_to_raise = exc
        except Exception as exc:
            self.fail("runtime_error", type(exc).__name__)
            if debug:
                failure_to_raise = exc
        self.trajectory["finished_at"] = utcnow()
        self.trajectory["usage"] = dict(self.counts, elapsed_seconds=round(time.monotonic() - self.started, 3))
        self.event("run_finished", {"final": self.trajectory["final"], "usage": self.trajectory["usage"]})
        self.save()
        if failure_to_raise is not None:
            # Collection must stop too; otherwise Ctrl-C would start the next
            # billable attempt after this one was merely marked interrupted.
            raise failure_to_raise
        return copy.deepcopy(self.trajectory)

    def remaining(self) -> float:
        """Return the next request's timeout or raise a hard run/token limit."""
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExceeded("max_seconds")
        if self.counts["reported_total_tokens"] >= self.config.max_reported_tokens:
            raise BudgetExceeded("max_reported_tokens")
        return min(remaining, self.config.request_timeout)

    def begin_messages(self, native: list[dict[str, Any]], tools: list[dict[str, Any]]) -> None:
        self.trajectory["tools"] = copy.deepcopy(tools)
        for message in native:
            self.append_message(message)
        self.save()

    def append_message(self, message: dict[str, Any]) -> str:
        item = copy.deepcopy(message)
        item["message_id"] = "m" + str(len(self.trajectory["messages"]) + 1).zfill(4)
        self.trajectory["messages"].append(item)
        return item["message_id"]

    def before_model_call(self) -> None:
        native = project_messages(self.trajectory["messages"])
        if len(json.dumps(native, ensure_ascii=False)) > self.config.max_context_characters:
            raise BudgetExceeded("max_context_characters")
        timeout = self._begin_call("actor")
        expected_images = {artifact["path"]: artifact["sha256"] for artifact in self.trajectory["artifacts"]
                           if artifact["kind"] in {"generated_image", "reference_image"}}
        self.actor.bind_request(self.root, timeout, expected_images=expected_images)

    def after_model_call(self, metadata: dict[str, Any] | None) -> None:
        self._end_call(metadata)

    def _begin_call(self, kind: str) -> float:
        timeout = self.remaining()
        if self.counts["provider_calls"] >= self.config.max_provider_calls:
            raise BudgetExceeded("max_provider_calls")
        if kind == "image" and self.counts["image_calls"] >= self.config.max_image_calls:
            raise BudgetExceeded("max_image_calls")
        if kind == "search" and self.counts["search_calls"] >= self.config.max_search_calls:
            raise BudgetExceeded("max_search_calls")
        self.counts["provider_calls"] += 1
        if kind in {"actor", "image", "search"}:
            self.counts[kind + "_calls"] += 1
        self.current_call = {"call_number": self.counts["provider_calls"], "kind": kind}
        self.event("provider_call_started", dict(self.current_call, timeout_seconds=timeout))
        return timeout

    def _end_call(self, metadata: dict[str, Any] | None) -> None:
        metadata = metadata or {}
        self._account_usage(metadata)
        self.event("provider_call_finished", dict(self.current_call or {}, metadata=metadata))
        self.current_call = None

    def _account_usage(self, metadata: dict[str, Any] | None) -> None:
        """Count one completed/failed call, including responses rejected by parsing.

        Missing usage includes requests stopped during input preparation after a
        call slot was reserved. Call counters are budget attempts, not a billing
        assertion; only actual provider-reported tokens contribute token totals.
        """
        usage = metadata.get("usage", {}) if isinstance(metadata, dict) else {}
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        if isinstance(tokens, (int, float)) and not isinstance(tokens, bool) and math.isfinite(tokens) and tokens >= 0:
            self.counts["reported_total_tokens"] += tokens
        else:
            self.counts["calls_without_token_usage"] += 1

    def start_step(self, number: int, context: list[dict[str, Any]], completion: dict[str, Any],
                   metadata: dict[str, Any] | None) -> None:
        """Persist the actual assistant response before executing its requested tool."""
        if context != project_messages(self.trajectory["messages"]):
            raise ProviderError("model_context_archive_mismatch")
        context_ids = [message["message_id"] for message in self.trajectory["messages"]]
        completion_id = self.append_message(completion)
        self.current_step = {"step_id": "s" + str(number).zfill(3), "context": copy.deepcopy(context),
                             "completion": [copy.deepcopy(completion)], "context_message_ids": context_ids,
                             "completion_message_ids": [completion_id], "actor_metadata": metadata,
                             "result": {"status": "running"}, "result_messages": []}
        self.trajectory["steps"].append(self.current_step)
        self.event("assistant_message", {"step_id": self.current_step["step_id"], "message_id": completion_id})
        self.save()

    def draw(self, prompt: str) -> dict[str, Any]:
        """Generate one image, verify/hash it, then expose it to the agent loop."""
        from PIL import Image
        timeout = self._begin_call("image")
        destination = self.root / "images" / (self.current_step["step_id"] + "_draw.png")
        metadata = self.image_provider.generate(prompt, destination, timeout=timeout)
        self._end_call(metadata)
        with Image.open(destination) as image:
            image.verify()
            width, height = image.size
        artifact = {"path": destination.relative_to(self.root).as_posix(), "sha256": sha256_file(destination),
                    "kind": "generated_image", "width": width, "height": height, "mime_type": "image/png"}
        previous = next((item for item in reversed(self.trajectory["artifacts"])
                         if item["kind"] == "generated_image"), None)
        runtime_state = draw_runtime_state(
            used=self.counts["image_calls"], limit=self.config.max_image_calls,
            same_prompt=self._last_draw_prompt == prompt,
            same_image=previous is not None and previous["sha256"] == artifact["sha256"],
        )
        self.trajectory["artifacts"].append(artifact)
        self._last_draw_prompt = prompt
        self.current_image = artifact["path"]
        self.event("artifact_created", artifact)
        self.current_step["image_metadata"] = metadata
        self.save()
        self.remaining()
        result = {"status": "ok", "image_path": artifact["path"], "sha256": artifact["sha256"],
                  "width": width, "height": height}
        if self.config.expose_runtime_state:
            result["runtime_state"] = runtime_state
        if self.research_enabled:
            result["runtime_state"] = dict(runtime_state, **self.research_runtime_state())
        return result

    def research_runtime_state(self) -> dict[str, Any]:
        return {"knowledge_calls_used": self.counts["knowledge_calls"],
                "remaining_knowledge_calls": self.config.max_knowledge_calls - self.counts["knowledge_calls"],
                "search_calls_used": self.counts["search_calls"],
                "remaining_search_calls": self.config.max_search_calls - self.counts["search_calls"],
                "remaining_image_calls": self.config.max_image_calls - self.counts["image_calls"],
                "remaining_provider_calls": self.config.max_provider_calls - self.counts["provider_calls"],
                "remaining_steps": self.config.max_steps - len(self.trajectory["steps"]),
                "generated_image_available": self.current_image is not None}

    def _record_research_result(self, name: str, result: dict[str, Any]) -> dict[str, Any]:
        """Commit actual tool evidence before exposing it in an actor observation."""
        if not isinstance(result, dict) or result.get("status", "ok") not in {"ok", "empty", "error"}:
            raise ProviderError("invalid_research_result")
        result = copy.deepcopy(result)
        result.setdefault("status", "ok")
        if result.get("tool", name) != name:
            raise ProviderError("research_tool_identity_mismatch")
        result["tool"] = name
        result["runtime_state"] = self.research_runtime_state()
        destination = self.root / "research" / (self.current_step["step_id"] + "_" + name + ".json")
        write_json(destination, result)
        artifact = {"path": destination.relative_to(self.root).as_posix(), "sha256": sha256_file(destination),
                    "kind": "research_result", "mime_type": "application/json", "tool_name": name,
                    "source_step_id": self.current_step["step_id"]}
        self.trajectory["artifacts"].append(artifact)
        self.event("artifact_created", artifact)
        record = {"path": artifact["path"], "sha256": artifact["sha256"]}
        self.current_step["research_result"] = {"status": "ok", "tool_status": result["status"], "tool_name": name,
            "result_path": record["path"], "result_sha256": record["sha256"]}
        if name == "image_search":
            self.current_step["research_result"]["reference_image_paths"] = [
                item["image_path"] for item in result["references"]]
        self.event("research_tool_finished", {"step_id": self.current_step["step_id"],
                                               **self.current_step["research_result"]})
        self.save()
        self.remaining()
        return dict(result, record=record)

    def knowledge_lookup(self, category_ids: list[str], query: str = "", max_results: int = 3) -> dict[str, Any]:
        if not self.research_enabled:
            raise ProviderError("research_tools_disabled")
        self.remaining()
        if self.counts["knowledge_calls"] >= self.config.max_knowledge_calls:
            raise BudgetExceeded("max_knowledge_calls")
        self.counts["knowledge_calls"] += 1
        self.event("research_tool_started", {"tool_name": "knowledge_lookup", "step_id": self.current_step["step_id"],
                                              "call_number": self.counts["knowledge_calls"]})
        result = self.knowledge_lookup_callback(category_ids, query=query, max_results=max_results)
        return self._record_research_result("knowledge_lookup", result)

    def image_search(self, query: str, max_results: int = 2) -> dict[str, Any]:
        """Research images never change current_image or consume a draw slot."""
        from PIL import Image
        if not self.research_enabled:
            raise ProviderError("research_tools_disabled")
        timeout = self._begin_call("search")
        result = self.reference_search.search(query, self.root, self.counts["search_calls"], timeout, max_results=max_results)
        if not isinstance(result, dict):
            raise ProviderError("invalid_reference_search_result")
        self._end_call(result.get("metadata"))
        references = result.get("references")
        if not isinstance(references, list) or len(references) > max_results:
            raise ProviderError("invalid_reference_search_result")
        for reference in references:
            if not isinstance(reference, dict):
                raise ProviderError("invalid_reference_search_result")
            try:
                path = safe_media_path(self.root, reference["image_path"])
                if (not path.is_relative_to(self.root / "references")
                        or path.relative_to(self.root).as_posix() != reference["image_path"]) or any(
                        existing["path"] == reference["image_path"] for existing in self.trajectory["artifacts"]):
                    raise ProviderError("invalid_reference_image_path")
                digest = sha256_file(path)
                if digest != reference["sha256"]:
                    raise ProviderError("reference_image_hash_mismatch")
                with Image.open(path) as image:
                    image.verify()
                    width, height = image.size
                    mime = Image.MIME.get(image.format, "application/octet-stream")
                provenance = {key: reference[key] for key in ("source_url", "landing_page", "provider", "source_id")}
                provenance.update({key: reference[key] for key in (
                    "creator", "attribution", "license", "license_url", "license_unknown",
                    "license_evidence", "photographic_origin_verified", "role") if key in reference})
            except (KeyError, TypeError, ValueError, OSError):
                raise ProviderError("invalid_reference_image_artifact") from None
            artifact = {"path": reference["image_path"], "sha256": digest, "kind": "reference_image",
                        "width": width, "height": height, "mime_type": mime,
                        "source_step_id": self.current_step["step_id"], "provenance": provenance}
            self.trajectory["artifacts"].append(artifact)
            self.event("artifact_created", artifact)
        return self._record_research_result("image_search", result)

    def finish_step(self, result_messages: list[dict[str, Any]]) -> None:
        self.current_step["result_messages"] = copy.deepcopy(result_messages)
        self.current_step["result_message_ids"] = [self.append_message(message) for message in result_messages]
        self.current_step["result"] = self.current_step.pop("research_result", None) or {
            "status": "ok", "image_path": self.current_image}
        self.event("tool_result", {"step_id": self.current_step["step_id"], "result": self.current_step["result"]})
        self.save()

    def submit(self) -> None:
        """Accept the latest complete draw output for pending human review."""
        self.remaining()
        if self.current_image is None:
            self.fail("no_image", "assistant_finished_without_drawing")
            return
        expected = next(artifact["sha256"] for artifact in self.trajectory["artifacts"]
                        if artifact["kind"] == "generated_image" and artifact["path"] == self.current_image)
        if sha256_file(self.root / self.current_image) != expected:
            raise ProviderError("submitted_image_hash_changed")
        self.current_step["result"] = {"status": "submitted"}
        self.trajectory["final"] = {"status": "submitted", "stop_reason": "assistant_submitted_latest_image",
            "image_path": self.current_image, "image_sha256": expected,
            "assistant_message_id": self.current_step["completion_message_ids"][0],
            "evaluation": {"status": "not_run"}, "human_review_status": "pending"}
        self.save()

    def _external_evaluation(self, task: dict[str, Any]) -> None:
        """Optional final-only judge; its result never becomes a model observation."""
        timeout = self._begin_call("external_evaluator")
        result = self.final_evaluator(task, self.current_image, self.root, timeout)
        if isinstance(result, dict):
            self._end_call(result.get("metadata", {}))
        try:
            validate_evaluation(result, task)
        except ValueError as error:
            raise ProviderError(str(error)) from None
        self.remaining()
        self.trajectory["final"]["evaluation"] = result
        self.trajectory["final"]["status"] = "success" if result["status"] == "passed" else "rejected_by_judge"
        self.trajectory["final"]["stop_reason"] = "external_evaluation_" + result["status"]

    def budget_exhausted(self, reason: str) -> None:
        self.fail("budget_exhausted", reason)

    def fail(self, status: str, code: str, **extra: Any) -> None:
        if self.current_call:
            provider = {"actor": self.actor, "image": self.image_provider,
                        "search": self.reference_search}.get(self.current_call["kind"])
            metadata = getattr(provider, "last_metadata", None)
            self._account_usage(metadata)
            self.event("provider_call_failed", dict(self.current_call, code=code, metadata=metadata or {}, **extra))
            self.current_call = None
        if self.current_step and self.current_step["result"]["status"] == "running":
            self.current_step["result"] = {"status": status, "error": {"code": code}}
        self.trajectory["final"] = {"status": status, "stop_reason": code, "image_path": self.current_image,
                                    "evaluation": {"status": "not_run"}, "human_review_status": "pending", **extra}

    def event(self, name: str, data: dict[str, Any]) -> None:
        self.event_number += 1
        value = {"event_id": self.event_number, "time": utcnow(), "event": name, "data": data}
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def save(self) -> None:
        write_json(self.root / "trajectory.json", self.trajectory)
