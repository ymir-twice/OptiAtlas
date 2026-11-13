"""Prompt-only benchmark generation, with isolated archives and resumable shards.

This is deliberately separate from training trajectory collection: benchmark
tasks always have split=test, no judge data is passed to the actor, and failed
tasks remain in the manifest denominator even when they produced no image.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path

from .logging import atomic_json, fingerprint, redact, utcnow


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prompts(path):
    """Retain only generation inputs and identifiers, never hidden judge fields."""
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError("Benchmark prompts must be a nonempty JSON list")
    records, identifiers, indices = [], set(), set()
    for position, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError("Benchmark prompt records must be objects")
        sample_id, prompt = value.get("sample_id"), value.get("generation_prompt")
        index, category = value.get("index", position), value.get("category_id", "uncategorized")
        if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sample_id):
            raise ValueError("sample_id must be a safe, nonempty file identifier")
        if type(index) is not int or index < 0:
            raise ValueError("Benchmark index must be a nonnegative integer")
        if sample_id in identifiers or index in indices:
            raise ValueError("Benchmark sample_id and index must be unique")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Each benchmark record needs a generation_prompt")
        if not isinstance(category, str) or not category.strip():
            raise ValueError("category_id must be a nonempty string")
        identifiers.add(sample_id)
        indices.add(index)
        records.append({"sample_id": sample_id, "index": index, "category_id": category,
                        "generation_prompt": prompt})
    return records


def _task(record):
    return {"task_id": record["sample_id"], "prompt": record["generation_prompt"],
            "scene_family": record["category_id"], "split": "test"}


def _implementation():
    root = Path(__file__).resolve().parent.parent
    # Research actions execute nested tools and read the knowledge cards. Bind
    # both to the experiment, so a changed knowledge base cannot silently alter
    # later samples in a resumed benchmark.
    paths = list((root / "optics_agent").rglob("*.py"))
    paths += list((root / "optics_agent" / "knowledge").rglob("*.json"))
    paths += [root / "optics_experiments" / name for name in (
        "benchmark_generate.py", "image_provider.py", "sensenova_actor.py",
        "gpt_image_provider.py", "wavespeed_image_provider.py", "company_api.py", "protocols.py", "agent_data.py")]
    return {str(path.relative_to(root)): _sha(path) for path in sorted(set(paths))
            if path.is_file() and not path.name.startswith("test_")
            and "tests" not in path.relative_to(root).parts}


def _model_identity(image):
    """Bind local checkpoint config and weight metadata without hashing 30GB."""
    model_path = image.get("model_path")
    if model_path is None or model_path == "":
        # Path("") resolves to cwd: remote providers have no local checkpoint,
        # and must never fingerprint unrelated training data or credentials.
        return {"path": None, "local": False}
    root = Path(model_path).expanduser()
    if not root.is_dir():
        return {"path": str(root.resolve()), "exists": False}
    metadata = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith(".") or ".cache" in path.parts:
            continue
        if path.suffix not in {".json", ".safetensors", ".bin", ".py", ".model", ".jinja", ".txt"}:
            continue
        stat = path.stat()
        record = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path.suffix in {".json", ".py", ".jinja", ".txt"}:
            record["sha256"] = _sha(path)
        metadata[path.relative_to(root).as_posix()] = record
    return {"path": str(root.resolve()), "files": metadata}


def _spec(values, records):
    configuration = {name: copy.deepcopy(values.get(name, {}))
                     for name in ("architecture", "image", "budget")}
    native_actor = values["architecture"] in {"qwen_sensenova", "qwen_draw_align", "native_api"}
    if values["architecture"] not in {"direct_draw_align", "direct_api"}:
        actor_key = "actor" if native_actor else "unified_actor"
        configuration[actor_key] = copy.deepcopy(values.get(actor_key, {}))
    if "research_tools" in values:
        configuration["research_tools"] = copy.deepcopy(values["research_tools"])
    # CUDA visibility and ports are deployment details, but model/generation
    # settings and the original prompt list remain part of the experiment.
    configuration["image"].pop("device", None)
    external_code = {}
    eval_repo_value = values.get("image", {}).get("eval_repo")
    if eval_repo_value:
        eval_repo = Path(eval_repo_value).expanduser()
        client = eval_repo / "engine/client/sensenova_u1_client.py"
        for source in [client, *sorted((eval_repo / "engine/client/modeling/sensenova_u1").rglob("*.py"))]:
            if source.is_file():
                external_code[source.relative_to(eval_repo).as_posix()] = _sha(source)
    public = redact(configuration)
    # Public logs strip URL query strings. Hash the original settings as well:
    # changing a legitimate api-version must invalidate resume even though its
    # redacted display URL remains identical. Only the digest is persisted.
    identity = {"config": public, "configuration_sha256": fingerprint(configuration),
                "prompts_sha256": fingerprint(records),
                "implementation": _implementation(), "external_code": external_code,
                "checkpoint": _model_identity(values.get("image", {}))}
    if values.get("image", {}).get("provider") in {"draw_align_fm", "draw_align_checkpoint"}:
        # Bind the exact selected training checkpoint, not just the base model
        # path or an unversioned exported adapter alias.
        if values["image"]["provider"] == "draw_align_fm":
            from .fm_image_provider import LocalFMCheckpointProvider as CheckpointProvider
        else:
            from .checkpoint_image_provider import LocalCheckpointProvider as CheckpointProvider
        identity["finetuned_checkpoint"] = CheckpointProvider(
            **{key: value for key, value in values["image"].items() if key != "provider"}
        ).identity
        base_model = identity["finetuned_checkpoint"].get("base_model")
        if base_model:
            identity["checkpoint"] = _model_identity({"model_path": base_model})
        root = Path(__file__).resolve().parent.parent
        fm_sources = [root / "optics_experiments/fm_image_provider.py",
                      root / "optics_experiments/checkpoint_image_provider.py"]
        fm_sources += sorted((root / "Draw-Align-SFT/draw_align_sft/models").glob("*.py"))
        fm_sources += sorted((root / "Draw-Align-SFT/draw_align_sft/legacy_sensenova").glob("*.py"))
        identity["finetuned_implementation"] = {
            str(path.relative_to(root)): _sha(path) for path in fm_sources if path.is_file()
        }
    if native_actor:
        # This records the intended served weights; HTTP alias alone cannot
        # prove which checkpoint an independently managed server actually loaded.
        actor_checkpoint = values.get("actor_checkpoint")
        identity["actor_checkpoint"] = (_model_identity({"model_path": actor_checkpoint})
                                         if actor_checkpoint else None)
        if values.get("actor_adapter"):
            identity["actor_adapter"] = _model_identity({"model_path": values["actor_adapter"]})
    return {"schema_version": 1, "execution_fingerprint": fingerprint(identity),
            "shard_index": values.get("shard_index", 0), "num_shards": values.get("num_shards", 1),
            "identity": identity}


def _seed(base, sample_id, attempt):
    # Independent of shard placement, ordering, pilot size and resume timing.
    if type(base) is not int or not 0 <= base <= 2147483647:
        raise ValueError("Image seed must be an integer in [0, 2147483647]")
    return (base + int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) + attempt) % 2147483648


def _make_provider(values):
    from .image_provider import make_image_provider
    return make_image_provider(values["image"])


def _apply_seed(provider, seed, image_settings):
    """Do not imply reproducible noise for an API that has no seed control."""
    identity = getattr(provider, "identity", {})
    supported = (image_settings.get("provider") != "gpt_images"
                 and identity.get("seed_controlled", True) is not False)
    if supported:
        provider.seed = seed
    return supported


def _make_runner(values, provider, seed):
    if values["architecture"] in {"direct_draw_align", "direct_api"}:
        return DirectImageRunner(provider, values.get("budget", {}).get("request_timeout", 600))
    from optics_agent.runner import AgentRunner, RunConfig
    if values["architecture"] in {"qwen_sensenova", "qwen_draw_align", "native_api"}:
        from optics_agent.llm import NativeChatModel
        actor = NativeChatModel(**values["actor"])
    else:
        from .sensenova_actor import LocalSenseNovaActor
        actor_config = dict(values.get("unified_actor", {}))
        actor_config["seed"] = seed
        actor = LocalSenseNovaActor(provider, **actor_config)
    return AgentRunner(actor, provider, RunConfig(**values.get("budget", {}), live_enabled=True),
                       research_tools=values.get("research_tools"))


class DirectImageRunner:
    """One native T2I call on the original prompt, with no Agent or rewrite.

    The shared archive filename is trajectory.json for exporter compatibility;
    its distinct schema/mode makes it ineligible for Agent SFT admission.
    """

    def __init__(self, provider, timeout):
        self.provider, self.timeout = provider, timeout

    def run(self, task, output, is_demo=False):
        if is_demo or task.get("split") != "test":
            raise ValueError("Direct benchmark generation requires a real test task")
        from PIL import Image
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        image = output / "images/draw.png"
        started = utcnow()
        context = getattr(self.provider, "set_request_context", None)
        if callable(context):
            context(task["task_id"])
        metadata = self.provider.generate(task["prompt"], image, timeout=self.timeout)
        with Image.open(image) as pixels:
            pixels.verify()
        digest = _sha(image)
        record = {"schema_version": "direct_benchmark_v1", "generation_mode": "direct",
                  "task": task, "is_demo": False, "started_at": started, "finished_at": utcnow(),
                  "provider": redact(metadata), "usage": {"image_calls": 1, "actor_calls": 0},
                  "artifacts": [{"path": "images/draw.png", "kind": "generated_image", "sha256": digest}],
                  "final": {"status": "submitted", "stop_reason": "direct_generation_completed",
                            "image_path": "images/draw.png", "image_sha256": digest}}
        atomic_json(output / "trajectory.json", record)
        return record


def _is_generated_image(artifact):
    """Classify current artifacts, while preserving the old image-only schema."""
    if artifact.get("kind") is not None:
        return artifact["kind"] == "generated_image"
    # Early benchmark archives did not record kinds. Their draw outputs were
    # exclusively under images/; references/ and research/ cannot be targets.
    path = Path(artifact.get("path", ""))
    return len(path.parts) >= 2 and path.parts[0] == "images" and path.suffix.lower() in {
        ".png", ".jpg", ".jpeg", ".webp"}


def _file_inventory(directory):
    return {path.relative_to(directory).as_posix(): _sha(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _check_saved(output, previous):
    for relative, expected in previous.get("attempt_ledger", {}).items():
        path = output / relative
        if not path.resolve().is_relative_to(output):
            raise ValueError("Archived attempt path escapes the benchmark output")
        if not path.is_dir() or _file_inventory(path) != expected:
            raise ValueError("Existing benchmark attempt changed or disappeared: " + relative)
    for row in previous.get("records", []):
        if row.get("image_path"):
            path = output / row["image_path"]
            if not path.resolve().is_relative_to(output):
                raise ValueError("Exported image path escapes the benchmark output")
            if path.exists() and _sha(path) != row["image_sha256"]:
                raise ValueError("Exported benchmark image changed: " + row["sample_id"])


def outcome_flags(row):
    """Describe execution separately from whether the last image can be scored."""
    status = row["status"]
    nonterminal = {"pending", "running"}
    technical = {"technical_failure", "runtime_error", "no_image", "interrupted"}
    if status not in nonterminal | technical | {"submitted", "budget_exhausted"}:
        raise ValueError("Unknown benchmark execution status: " + str(status))
    return {"budget_limited": status == "budget_exhausted",
            "technical_failed": status in technical,
            "not_submitted": status not in nonterminal | {"submitted"},
            "image_available": bool(row.get("image_path"))}


def outcome_counts(rows, *, total=None):
    """Missing worker manifests are pending, never completed or failed."""
    flags = [outcome_flags(row) for row in rows]
    completed = sum(row["status"] not in {"pending", "running"} for row in rows)
    not_submitted = sum(flag["not_submitted"] for flag in flags)
    images = sum(flag["image_available"] for flag in flags)
    return {"completed": completed, "pending": (len(rows) if total is None else total) - completed,
            "submitted": sum(row["status"] == "submitted" for row in rows),
            "budget_limited": sum(flag["budget_limited"] for flag in flags),
            "technical_failed": sum(flag["technical_failed"] for flag in flags),
            "not_submitted": not_submitted,
            "failed": not_submitted,  # Legacy name: includes budget exhaustion, not just technical errors.
            "images": images, "exported": images}


def _scan(output, records, spec, *, cached=None, refresh_ids=None):
    # Revalidate every archive once on resume. Within a locked worker only the
    # current sample changes, so avoid rehashing the whole image corpus per item.
    rows = []
    ledger = copy.deepcopy(cached.get("attempt_ledger", {})) if cached else {}
    cached_rows = {row["sample_id"]: row for row in cached["records"]} if cached else {}
    allowed = {record["sample_id"] for record in records}
    attempts_root = output / "attempts"
    if attempts_root.exists():
        unknown = {path.name for path in attempts_root.iterdir()} - allowed
        if unknown:
            raise ValueError("Unknown sample directories in benchmark archives")
    for record in records:
        if record["sample_id"] in cached_rows and record["sample_id"] not in (refresh_ids or set()):
            row = copy.deepcopy(cached_rows[record["sample_id"]])
            row.update(outcome_flags(row))
            rows.append(row)
            continue
        row = {**record, "status": "pending", "stop_reason": None, "usage": {},
               "success": False, "image_path": None,
               "image_sha256": None, "trajectory": None, "attempt": None, "attempts": []}
        sample_root = attempts_root / record["sample_id"]
        directories = sorted(sample_root.iterdir()) if sample_root.is_dir() else []
        for number, directory in enumerate(directories):
            if directory.name != f"attempt_{number:03d}" or not directory.is_dir():
                raise ValueError("Benchmark attempt sequence is invalid")
            marker_path = directory / "attempt.json"
            if not marker_path.is_file():
                raise ValueError("Attempt is missing its execution identity: " + str(directory))
            marker = json.loads(marker_path.read_text())
            if (marker.get("execution_fingerprint") != spec["execution_fingerprint"]
                    or marker.get("task") != _task(record) or marker.get("attempt") != number):
                raise ValueError("Existing attempt does not match benchmark/configuration")
            trajectory_path = directory / "run/trajectory.json"
            trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.is_file() else None
            status, image_path, image_hash = "interrupted", None, None
            stop_reason, usage = "incomplete_attempt_archive", {}
            if trajectory:
                if trajectory.get("task") != _task(record) or trajectory.get("is_demo") is not False:
                    raise ValueError("Archived trajectory task/mode differs from benchmark")
                final = trajectory.get("final", {})
                status = final.get("status", "interrupted")
                stop_reason, usage = final.get("stop_reason"), trajectory.get("usage", {})
                if status == "running" or not trajectory.get("finished_at"):
                    status = "interrupted"
                    stop_reason = "incomplete_attempt_archive"
                image_relative = final.get("image_path")
                if not image_relative and status == "interrupted" and trajectory.get("artifacts"):
                    image_relative = next((item["path"] for item in reversed(trajectory["artifacts"])
                                           if _is_generated_image(item)), None)
                if status == "submitted" and not image_relative:
                    raise ValueError("Submitted benchmark attempt has no final image")
                if image_relative:
                    source = (directory / "run" / image_relative).resolve()
                    if not source.is_relative_to((directory / "run").resolve()):
                        raise ValueError("Archived image must remain inside its attempt")
                    artifact = next((item for item in trajectory.get("artifacts", [])
                                     if item["path"] == image_relative), None)
                    if artifact and not _is_generated_image(artifact):
                        raise ValueError("A benchmark output must be a generated image, not a research artifact")
                    if not source.is_file() or not artifact or _sha(source) != artifact["sha256"]:
                        raise ValueError("Archived benchmark image is missing or changed")
                    image_path, image_hash = source, artifact["sha256"]
            elif (directory / "error.json").is_file():
                error = json.loads((directory / "error.json").read_text())
                status, stop_reason = error["status"], error.get("code")
            relative = directory.relative_to(output).as_posix()
            attempt = {"attempt": number, "status": status, "seed": marker["seed"],
                       "seed_applied": marker.get("seed_applied", True),
                       "started_at": marker.get("started_at"),
                       "directory": relative,
                       "trajectory": trajectory_path.relative_to(output).as_posix() if trajectory else None}
            row["attempts"].append(attempt)
            row.update({"status": status, "stop_reason": stop_reason, "usage": usage,
                        "success": status == "submitted", "attempt": number,
                        "trajectory": attempt["trajectory"], "image_sha256": image_hash,
                        "image_path": f"images/{record['sample_id']}.png" if image_path else None})
            # Only the latest attempt is exported. Older images remain immutable
            # in their own archives; never select the best attempt by judge score.
            row["_source_image"] = image_path
            ledger[relative] = _file_inventory(directory)
        source = row.pop("_source_image", None)
        destination = output / "images" / (record["sample_id"] + ".png")
        if source:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or _sha(destination) != row["image_sha256"]:
                temporary = destination.with_suffix(f".{os.getpid()}.tmp")
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
        elif destination.exists():
            destination.unlink()
        row.update(outcome_flags(row))
        rows.append(row)
    return rows, ledger


def _manifest(output, records, spec, values, selected_records, benchmark_count, *, cached=None, refresh_ids=None):
    rows, ledger = _scan(output, records, spec, cached=cached, refresh_ids=refresh_ids)
    counts = outcome_counts(rows)
    selected_ids = {record["sample_id"] for record in selected_records}
    selected_rows = [row for row in rows if row["sample_id"] in selected_ids]
    selected_counts = outcome_counts(selected_rows)
    failures = [row for row in selected_rows if row["not_submitted"]]
    latest_failure = max(failures, key=lambda row: (row["attempts"][-1].get("started_at") or ""), default=None)
    latest_issue = ({"sample_id": latest_failure["sample_id"], "status": latest_failure["status"],
                     "budget_limited": latest_failure["budget_limited"],
                     "technical_failed": latest_failure["technical_failed"],
                     "image_available": latest_failure["image_available"],
                     "stop_reason": latest_failure.get("stop_reason") or latest_failure["status"],
                     "trajectory": latest_failure["trajectory"],
                     "attempt_directory": latest_failure["attempts"][-1]["directory"]}
                    if latest_failure else None)
    summary = {"schema_version": 1, "execution_fingerprint": spec["execution_fingerprint"],
               "architecture": values["architecture"], "prompts_file": str(Path(values["prompts_file"]).resolve()),
               "benchmark_count": benchmark_count, "shard_index": spec["shard_index"],
               "num_shards": spec["num_shards"], "shard_total_count": len(records),
               "selected_count": len(selected_records), **counts,
               **{"selected_" + key: value for key, value in selected_counts.items()},
               "complete": counts["submitted"] == len(records),
               "status": "running", "updated_at": utcnow(), "records": rows, "attempt_ledger": ledger,
               "failure_stop_reasons": dict(Counter(row.get("stop_reason") or row["status"] for row in failures)),
               "latest_issue": latest_issue, "latest_failure": latest_issue,
               "note": "Generation only; submitted is not a benchmark score or proof of optical correctness. "
                       "Budget-limited images remain available for scoring without being marked submitted. "
                       "failed is a legacy alias for not_submitted, including budget_limited and technical_failed. "
                       "All tasks remain in the denominator."}
    atomic_json(output / "manifest.json", summary)
    return summary


def _progress(logger, name, **fields):
    if logger:
        logger.progress(name, **fields)
    else:
        print(name + " " + json.dumps(fields, ensure_ascii=False), flush=True)


def _technical_failure(row):
    """Trip on broken execution/protocol, not unsuccessful image refinement."""
    return row["status"] in {"technical_failure", "runtime_error", "no_image"}


def run(cfg, logger=None):
    values = copy.deepcopy(cfg["benchmark"])
    architecture = values.get("architecture")
    if architecture not in {"qwen_sensenova", "qwen_draw_align", "sensenova_unified", "native_api", "direct_draw_align", "direct_api"}:
        raise ValueError("Unknown benchmark architecture")
    image_provider = values.get("image", {}).get("provider")
    if architecture == "native_api" and image_provider not in {"openai_images", "gpt_images"}:
        raise ValueError("native_api benchmark workers require an openai_images or gpt_images provider")
    if architecture == "direct_api" and image_provider not in {"openai_images", "gpt_images", "wavespeed"}:
        raise ValueError("direct_api benchmark workers require a remote image provider")
    if architecture in {"qwen_draw_align", "direct_draw_align"} and image_provider not in {"draw_align_fm", "draw_align_checkpoint"}:
        raise ValueError("Draw-Align benchmark workers require a draw_align_fm or draw_align_checkpoint provider")
    if architecture in {"qwen_sensenova", "sensenova_unified"} and image_provider != "sensenova_local":
        raise ValueError("Benchmark workers require a local SenseNova image provider")
    research = values.get("research_tools")
    if research is not None:
        from optics_agent.runner import validate_research_config
        research = validate_research_config(research)
    if architecture == "sensenova_unified" and research and research.get("enabled", False):
        raise ValueError("Research tools require a native actor architecture")
    if "live_enabled" in values.get("budget", {}):
        raise ValueError("execute controls live execution, not budget.live_enabled")
    shard, shards = values.get("shard_index", 0), values.get("num_shards", 1)
    limit = values.get("max_samples")
    failure_limit = values.get("max_consecutive_failures", 3)
    if type(failure_limit) is not int or failure_limit < 0:
        raise ValueError("max_consecutive_failures must be a nonnegative integer (0 disables the guard)")
    if type(shard) is not int or type(shards) is not int or shards < 1 or not 0 <= shard < shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_samples must be null or a positive integer")
    all_records = read_prompts(values["prompts_file"])
    records = [record for position, record in enumerate(all_records) if position % shards == shard]
    selected = [record for position, record in enumerate(all_records)
                if position % shards == shard and (limit is None or position < limit)]
    output = Path(cfg["output_dir"]).expanduser().resolve()
    spec = _spec(values, all_records)
    if not values.get("execute", True):
        return {"status": "plan", "architecture": architecture, "benchmark_count": len(all_records),
                "shard_total_count": len(records), "selected_count": len(selected),
                "sample_ids": [record["sample_id"] for record in selected], "output_dir": str(output),
                "execution_fingerprint": spec["execution_fingerprint"],
                "max_image_calls": values.get("budget", {}).get("max_image_calls", 3)}
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process already owns this benchmark shard") from None
        spec_path, manifest_path = output / "benchmark_spec.json", output / "manifest.json"
        if spec_path.exists():
            if not values.get("resume", True):
                raise FileExistsError("Benchmark output exists; enable resume or use another output_dir")
            if json.loads(spec_path.read_text()) != spec:
                raise ValueError("Cannot resume: benchmark, model, prompt, code or configuration changed")
        else:
            if manifest_path.exists() or (output / "attempts").exists():
                raise ValueError("Existing benchmark archives lack a resume identity")
            atomic_json(spec_path, spec)
        if manifest_path.exists():
            _check_saved(output, json.loads(manifest_path.read_text()))
        summary = _manifest(output, records, spec, values, selected, len(all_records))
        previous = {row["sample_id"]: row for row in summary["records"]}
        pending = [record for record in selected if previous[record["sample_id"]]["status"] == "pending"
                   or (values.get("retry_failed", False)
                       and previous[record["sample_id"]]["status"] != "submitted")]
        resumed, started = len(selected) - len(pending), time.monotonic()
        _progress(logger, "benchmark_started", shard=shard, total=len(selected), resumed=resumed,
                  pending=len(pending), architecture=architecture)
        provider, first_runner, phase = None, None, "startup"
        consecutive_failures = 0
        try:
            if pending:
                load_started = time.monotonic()
                _progress(logger, "benchmark_model_load_started", shard=shard, architecture=architecture)
                provider = _make_provider(values)
                first_record = pending[0]
                first_seed = _seed(values["image"].get("seed", 42), first_record["sample_id"],
                                   len(previous[first_record["sample_id"]]["attempts"]))
                _apply_seed(provider, first_seed, values["image"])
                # Validate actor imports, API-key settings and budgets before
                # consuming a benchmark attempt. The actor shares this provider.
                first_runner = _make_runner(values, provider, first_seed)
                load = getattr(provider, "_load", None)
                if callable(load):
                    load()
                prepare = getattr(getattr(first_runner, "actor", None), "prepare", None)
                if callable(prepare):
                    # Validate structured decoding before consuming any sample.
                    prepare()
                _progress(logger, "benchmark_model_load_finished", shard=shard, architecture=architecture,
                          elapsed_seconds=round(time.monotonic() - load_started, 1))
            phase = "generation"
            for completed, record in enumerate(pending):
                prior = previous[record["sample_id"]]
                attempt = len(prior["attempts"])
                seed = _seed(values["image"].get("seed", 42), record["sample_id"], attempt)
                seed_applied = _apply_seed(provider, seed, values["image"])
                destination = output / "attempts" / record["sample_id"] / f"attempt_{attempt:03d}"
                destination.mkdir(parents=True, exist_ok=False)
                atomic_json(destination / "attempt.json", {"execution_fingerprint": spec["execution_fingerprint"],
                            "task": _task(record), "attempt": attempt, "seed": seed,
                            "seed_applied": seed_applied, "started_at": utcnow()})
                _progress(logger, "benchmark_sample_started", shard=shard, sample_id=record["sample_id"],
                          completed=resumed + completed, total=len(selected), attempt=attempt)
                try:
                    runner = first_runner if completed == 0 else _make_runner(values, provider, seed)
                    runner.run(_task(record), destination / "run", is_demo=False)
                except KeyboardInterrupt:
                    atomic_json(destination / "error.json", {"status": "interrupted", "code": "KeyboardInterrupt"})
                    raise
                except Exception as error:
                    from optics_agent.providers import ProviderError
                    code = error.code if isinstance(error, ProviderError) else type(error).__name__
                    atomic_json(destination / "error.json", {"status": "runtime_error", "code": code,
                                "provider": redact(getattr(provider, "last_metadata", None))})
                finally:
                    summary = _manifest(output, records, spec, values, selected, len(all_records),
                                        cached=summary, refresh_ids={record["sample_id"]})
                row = next(row for row in summary["records"] if row["sample_id"] == record["sample_id"])
                consecutive_failures = consecutive_failures + 1 if _technical_failure(row) else 0
                elapsed = time.monotonic() - started
                _progress(logger, "benchmark_sample_finished", shard=shard, sample_id=record["sample_id"],
                          status=row["status"], stop_reason=row["stop_reason"],
                          trajectory=row["trajectory"], consecutive_technical_failures=consecutive_failures,
                          completed=resumed + completed + 1, total=len(selected),
                          submitted=summary["submitted"], budget_limited=summary["budget_limited"],
                          technical_failed=summary["technical_failed"], not_submitted=summary["not_submitted"],
                          images=summary["images"], pending=summary["selected_pending"],
                          elapsed_seconds=round(elapsed, 1),
                          eta_seconds=round(elapsed / (completed + 1) * (len(pending) - completed - 1), 1))
                if failure_limit and consecutive_failures >= failure_limit:
                    summary["stop_reason"] = "consecutive_technical_failures"
                    summary["failure_guard"] = {"limit": failure_limit, "consecutive_failures": consecutive_failures,
                                                "sample_id": row["sample_id"], "last_stop_reason": row["stop_reason"]}
                    _progress(logger, "benchmark_failure_guard_stopped", shard=shard,
                              **summary["failure_guard"], pending=summary["selected_pending"],
                              manifest=str(manifest_path), trajectory=row["trajectory"])
                    break
        except BaseException as error:
            interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
            summary["status"] = "interrupted" if interrupted else "failed"
            summary["exit_code"] = 130 if interrupted else 1
            summary["error"] = {"stage": phase, "code": type(error).__name__}
            atomic_json(manifest_path, summary)
            _progress(logger, "benchmark_interrupted" if interrupted else "benchmark_failed",
                      shard=shard, stage=phase, error_type=type(error).__name__, manifest=str(manifest_path))
            raise
        summary["exit_code"] = 1 if summary["selected_failed"] or summary["selected_pending"] else 0
        summary["status"] = "failed" if summary["exit_code"] else "completed"
        atomic_json(manifest_path, summary)
        _progress(logger, "benchmark_finished", shard=shard, submitted=summary["submitted"],
                  images=summary["images"], budget_limited=summary["budget_limited"],
                  technical_failed=summary["technical_failed"], not_submitted=summary["not_submitted"],
                  pending=summary["pending"], completed=summary["completed"],
                  manifest=str(manifest_path))
        return summary
