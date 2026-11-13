"""Independent GPU or remote-API workers, with progress and scorer export."""
from __future__ import annotations

import copy
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
from urllib.parse import urlsplit, urlunsplit

from omegaconf import OmegaConf
from PIL import Image, PngImagePlugin

from .benchmark_generate import outcome_counts, outcome_flags
from .logging import atomic_json, fingerprint


def resolve_devices(devices, environment=None):
    """Respect the scheduler's visible-device mapping, including GPU UUIDs."""
    environment = os.environ if environment is None else environment
    if not isinstance(devices, list) or not devices or any(type(i) is not int or i < 0 for i in devices):
        raise ValueError("benchmark_launcher.devices must be a nonempty list of distinct GPU indices")
    if len(set(devices)) != len(devices):
        raise ValueError("Each benchmark worker needs a distinct GPU")
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(i) for i in devices]
    mapping = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if max(devices) >= len(mapping) or any(entry == "-1" for entry in mapping):
        raise ValueError("benchmark_launcher.devices exceeds CUDA_VISIBLE_DEVICES; use indices within the allocation")
    selected = [mapping[i] for i in devices]
    if len(set(selected)) != len(selected):
        raise ValueError("CUDA_VISIBLE_DEVICES assigns the same GPU to multiple workers")
    return selected


def worker_count(cfg):
    """Remote API concurrency is independent of the node's GPU allocation."""
    options = cfg["benchmark_launcher"]
    if cfg["benchmark"]["architecture"] in {"native_api", "direct_api"}:
        count = options.get("worker_count")
        if type(count) is not int or count < 1:
            raise ValueError("benchmark_launcher.worker_count must be a positive integer for remote API architectures")
        return count
    devices = options.get("devices")
    # Validate the topology without consulting this machine's visible devices;
    # merge is also usable on a host other than the one that ran generation.
    return len(resolve_devices(devices, environment={}))


def _source(cfg):
    rows = json.loads(Path(cfg["benchmark"]["prompts_file"]).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("The benchmark prompts file must contain a nonempty list")
    seen = set()
    for row in rows:
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sid) or sid in seen:
            raise ValueError("Benchmark sample IDs must be unique safe filenames")
        if not isinstance(row.get("generation_prompt"), str) or not row["generation_prompt"].strip():
            raise ValueError("Every benchmark task needs its original generation_prompt")
        seen.add(sid)
    limit = cfg["benchmark"].get("max_samples")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("benchmark.max_samples must be a positive integer or null")
    return rows, min(limit or len(rows), len(rows))


def _read_shard_manifest(path, index, count):
    """Open once, without a stat/open race, and validate fields used by telemetry."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("shard_index") != index
            or value.get("num_shards") != count or not isinstance(value.get("records"), list)
            or not isinstance(value.get("execution_fingerprint"), str)
            or not isinstance(value.get("architecture"), str)):
        raise ValueError("Invalid shard manifest identity or records")
    seen = set()
    for row in value["records"]:
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str) or row["sample_id"] in seen:
            raise ValueError("Invalid or duplicate shard sample ID")
        seen.add(row["sample_id"])
        if not isinstance(row.get("status"), str):
            raise ValueError("Invalid shard outcome status")
        if row.get("stop_reason") is not None and not isinstance(row["stop_reason"], str):
            raise ValueError("Invalid shard stop reason")
        outcome_flags(row)
        usage = row.get("usage", {})
        if not isinstance(usage, dict) or any(type(usage.get(key, 0)) is not int or usage.get(key, 0) < 0
                                             for key in ("image_calls", "actor_calls")):
            raise ValueError("Invalid shard usage counters")
    issue = value.get("latest_issue") or value.get("latest_failure")
    if issue and (not isinstance(issue, dict)
                  or any(not isinstance(issue.get(key), str) for key in ("sample_id", "stop_reason", "attempt_directory"))
                  or issue["sample_id"] not in seen):
        raise ValueError("Invalid shard latest issue")
    if not isinstance(value.get("updated_at", ""), str) or (value.get("failure_guard") and not isinstance(value["failure_guard"], dict)):
        raise ValueError("Invalid shard update or failure guard")
    return value


def _manifest_read_error(error):
    return {"error_type": type(error).__name__, **({"errno": error.errno} if isinstance(error, OSError) else {})}


def _shard_manifests(root, count, *, required_shards=()):
    """Strict final snapshots: retry briefly, never substitute cached progress.

    A never-created shard may be pending in a standalone partial merge. Once a
    shard has an archive, or the coordinator has seen it, an unreadable manifest
    must stop publication before old exports could be removed.
    """
    manifests = []
    required = set(required_shards)
    for index in range(count):
        path = root / "shards" / f"shard_{index:02d}" / "manifest.json"
        value, error = None, None
        for attempt in range(6):
            try:
                value = _read_shard_manifest(path, index, count)
                break
            except (OSError, ValueError) as caught:
                error = caught
                if attempt < 5:
                    time.sleep(.1 * (attempt + 1))
        if value is None:
            if isinstance(error, FileNotFoundError) and index not in required and not path.parent.exists():
                continue
            raise RuntimeError(f"Cannot read final shard manifest after retries: {path} "
                               f"({type(error).__name__}); existing exports were not replaced") from error
        manifests.append((path.parent, value))
    return manifests


def _progress_manifests(root, count, cache):
    """Best-effort telemetry only; no filesystem hiccup may kill GPU workers."""
    manifests, stale, unavailable, errors = [], [], [], []
    for index in range(count):
        path = root / "shards" / f"shard_{index:02d}" / "manifest.json"
        value, error = None, None
        for attempt in range(2):
            try:
                value = _read_shard_manifest(path, index, count)
                break
            except (OSError, ValueError) as caught:
                error = caught
                if attempt == 0:
                    time.sleep(.05)
        if value is not None:
            cache[index] = value
        else:
            errors.append({"shard_index": index, **_manifest_read_error(error)})
            if index in cache:
                value = cache[index]
                stale.append(index)
            else:
                unavailable.append(index)
        if value is not None:
            manifests.append((path.parent, value))
    return manifests, stale, unavailable, errors


def _progress(root, count, selected_ids, *, cache=None):
    manifests, stale, unavailable, read_errors = _progress_manifests(root, count, cache if cache is not None else {})
    records = [row for _, manifest in manifests for row in manifest["records"]
               if row["sample_id"] in selected_ids]
    failures = [row for row in records if outcome_flags(row)["not_submitted"]]
    recent = []
    for _, manifest in manifests:
        issue = manifest.get("latest_issue") or manifest.get("latest_failure")
        if issue and issue["sample_id"] in selected_ids:
            row = next((row for row in records if row["sample_id"] == issue["sample_id"]), None)
            if row is None or not outcome_flags(row)["not_submitted"]:
                continue
            recent.append((manifest.get("updated_at", ""),
                           {**issue, "status": row["status"], **outcome_flags(row),
                            "shard_index": manifest["shard_index"]}))
    recent.sort(key=lambda item: item[0])
    counts = outcome_counts(records, total=len(selected_ids))
    latest_issue = recent[-1][1] if recent else None
    return {
        "total": len(selected_ids),
        **counts, "with_image": counts["images"],
        "draw_calls": sum(row.get("usage", {}).get("image_calls", 0) for row in records),
        "actor_calls": sum(row.get("usage", {}).get("actor_calls", 0) for row in records),
        "failure_stop_reasons": dict(Counter(row.get("stop_reason") or row["status"] for row in failures)),
        "latest_issue": latest_issue, "latest_failure": latest_issue,
        "stale_manifest_shards": stale, "unavailable_manifest_shards": unavailable,
        "manifest_read_errors": read_errors,
    }


def merge(cfg, *, required_shards=()):
    """Losslessly export all available final pixels; never drop failed tasks."""
    root = Path(cfg["output_dir"]).resolve()
    sources, selected_count = _source(cfg)
    count = worker_count(cfg)
    by_id = {row["sample_id"]: row for row in sources}
    gathered = {}
    stopped_workers = []
    execution_keys = set()
    for shard_root, manifest in _shard_manifests(root, count, required_shards=required_shards):
        if manifest.get("failure_guard"):
            stopped_workers.append({"shard_index": manifest["shard_index"], **manifest["failure_guard"]})
        if manifest["num_shards"] != count or manifest["architecture"] != cfg["benchmark"]["architecture"]:
            raise ValueError("Shard architecture/topology does not match the merge configuration")
        execution_keys.add(manifest["execution_fingerprint"])
        if len(execution_keys) > 1:
            raise ValueError("Cannot merge shards from different model/config/code fingerprints")
        for row in manifest["records"]:
            sid = row["sample_id"]
            if sid not in by_id or sid in gathered or row["generation_prompt"] != by_id[sid]["generation_prompt"]:
                raise ValueError("Shard identities/prompts do not match the benchmark")
            gathered[sid] = (shard_root, row, manifest["execution_fingerprint"])
    image_dir, result_dir = root / "images", root / "results"
    image_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sources:
        sid = source["sample_id"]
        row = {"sample_id": sid, "index": source["index"], "category_id": source["category_id"],
               "generation_prompt": source["generation_prompt"], "status": "pending", "success": False,
               "image_path": None, "image_sha256": None, "trajectory": None}
        generation_key = None
        if sid in gathered:
            shard_root, saved, execution_key = gathered[sid]
            row.update(saved)
            if saved.get("trajectory"):
                row["trajectory"] = str((shard_root / saved["trajectory"]).relative_to(root))
            if saved.get("image_path"):
                path = (shard_root / saved["image_path"]).resolve()
                if not path.is_relative_to(shard_root) or not path.is_file():
                    raise ValueError("Missing or escaping shard image: " + sid)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != saved["image_sha256"]:
                    raise ValueError("Shard image changed after generation: " + sid)
                generation_key = fingerprint({"sample_id": sid, "prompt": source["generation_prompt"],
                                              "execution": execution_key, "source_image_sha256": digest})
                destination = image_dir / (sid + ".png")
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("light_omni_generation", generation_key)
                temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
                with Image.open(path) as image:
                    image.load()
                    # PNG metadata lets eval-repo verify provenance without legacy-resume overrides.
                    image.save(temporary, format="PNG", pnginfo=metadata)
                temporary.replace(destination)
                row.update(source_image_sha256=digest, image_path=str(destination.relative_to(root)),
                           image_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
        if not row["image_path"]:
            # A failed new attempt must not leave an older, misleading final PNG.
            (image_dir / (sid + ".png")).unlink(missing_ok=True)
        row["image_selection"] = ("submitted" if row["success"] else "last_available_unsubmitted") if row["image_path"] else "none"
        row.update(outcome_flags(row))
        row["generation_fingerprint"] = generation_key
        rows.append(row)
        # Scoring may inspect a last available image even if the Agent did not submit.
        # Keep that distinction explicit; status here means a valid image artifact exists.
        atomic_json(result_dir / (sid + ".json"), {
            "sample_id": sid, "index": source["index"], "category_id": source["category_id"],
            "generation_prompt": source["generation_prompt"],
            "model_name": cfg["benchmark"]["architecture"],
            "status": "success" if row["image_path"] else "failed",
            "agent_status": row["status"], "agent_submitted": row["success"],
            **outcome_flags(row),
            "image_selection": row["image_selection"], "generation_fingerprint": generation_key,
            "generated_image_path": str(root / row["image_path"]) if row["image_path"] else None,
            "image_sha256": row["image_sha256"], "trajectory": row["trajectory"],
        })
    selected = rows[:selected_count]
    selected_counts = outcome_counts(selected)
    failed, pending = selected_counts["not_submitted"], selected_counts["pending"]
    result = {
        "schema_version": 1, "architecture": cfg["benchmark"]["architecture"],
        "benchmark_count": len(rows), "selected_count": selected_count,
        **outcome_counts(rows),
        **{"selected_" + key: value for key, value in selected_counts.items()},
        "complete": all(r["success"] for r in rows),
        "status": "failed" if failed else "incomplete" if pending else "completed",
        "exit_code": 1 if failed or pending else 0,
        "failure_stop_reasons": dict(Counter(r.get("stop_reason") or r["status"] for r in selected
                                             if r["not_submitted"])),
        "stopped_workers": stopped_workers,
        "note": "Budget-limited images are exported and can be scored; their agents did not submit. "
                "failed is a legacy alias for not_submitted, including budget_limited and technical_failed. "
                "All benchmark tasks remain in the denominator.",
        "records": rows,
    }
    atomic_json(root / "manifest.json", result)
    summary = {k: v for k, v in result.items() if k != "records"}
    summary.update(manifest=str(root / "manifest.json"), image_dir=str(image_dir), results=str(result_dir))
    atomic_json(root / "summary.json", summary)
    return summary


def uses_remote_actor(benchmark):
    """Allow the explicit Gemini/local-generator profile without a local probe.

    A flag alone must not bypass local-model readiness or send credentials to
    an arbitrary endpoint. Keep this narrow until another remote policy is added.
    """
    remote = benchmark.get("actor_remote", False)
    if type(remote) is not bool:
        raise ValueError("benchmark.actor_remote must be boolean")
    if not remote:
        return False
    actor = benchmark.get("actor", {})
    if (not isinstance(actor, dict) or benchmark.get("architecture") != "qwen_draw_align"
            or actor.get("chat_completions_url") != "https://api.example.invalid/v1/chat/completions"
            or actor.get("model") != "google/gemini-3.6-flash"
            or actor.get("api_key_env") != "WAVESPEED_API_KEY" or actor.get("auth") != "bearer"
            or actor.get("include_model") is not True
            or benchmark.get("actor_checkpoint") or benchmark.get("actor_adapter")):
        raise ValueError("Remote local-generator policy must use the configured WaveSpeed Gemini endpoint without local weights")
    return True


def _check_actor(cfg):
    if uses_remote_actor(cfg["benchmark"]):
        key = os.environ.get(cfg["benchmark"]["actor"]["api_key_env"], "").strip()
        if not key or key == "local-benchmark":
            raise ValueError("Remote Gemini policy credential is missing; no workers started")
        return
    if cfg["benchmark"]["architecture"] not in {"qwen_sensenova", "qwen_draw_align"}:
        return
    import requests
    actor = cfg["benchmark"]["actor"]
    parts = urlsplit(actor["chat_completions_url"])
    url = urlunsplit(parts._replace(path=parts.path.removesuffix("/chat/completions") + "/models", query=""))
    key = os.environ.get(actor.get("api_key_env", "OPTICS_TEACHER_API_KEY"), "local-benchmark")
    try:
        response = requests.get(url, headers={"Authorization": "Bearer " + key}, timeout=10)
        response.raise_for_status()
        models = {row["id"] for row in response.json()["data"]}
    except Exception as error:
        raise RuntimeError("Qwen actor service is not ready; start serve_qwen_agent.sh first (" + type(error).__name__ + ")") from None
    if actor["model"] not in models:
        raise ValueError("Qwen service served-model-name does not match benchmark.actor.model")


def _stop(children):
    for process in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 8
    for process in children:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)


def run(cfg, logger=None):
    """One coordinator; each child owns an API/GPU worker and one durable shard."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Launch once per node, not with torchrun")
    options = cfg["benchmark_launcher"]
    remote_actor = uses_remote_actor(cfg["benchmark"])
    count = worker_count(cfg)
    remote_api = cfg["benchmark"]["architecture"] in {"native_api", "direct_api"}
    devices = [None] * count if remote_api else resolve_devices(options["devices"])
    sources, selected_count = _source(cfg)
    root = Path(cfg["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".benchmark-launch.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError("Another benchmark launcher owns this output directory") from None
    children, streams = [], []
    interrupted = False
    coordinator_error = None
    progress_cache = {}
    previous_handlers = {}
    def stop_on_signal(number, frame):
        raise SystemExit(128 + number)
    try:
        for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            previous_handlers[number] = signal.signal(number, stop_on_signal)
        launch_dir = root / "launches" / uuid.uuid4().hex
        launch_dir.mkdir(parents=True)
        plan = []
        for index, device in enumerate(devices):
            worker = copy.deepcopy(cfg)
            worker.pop("benchmark_launcher")
            worker.update(task="benchmark_generate", output_dir=str(root / "shards" / f"shard_{index:02d}"), run_id=None)
            worker["benchmark"].update(shard_index=index, num_shards=count)
            worker.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
            worker["hydra"] = {"job": {"chdir": False}, "run": {"dir": "${output_dir}/hydra/${now:%Y%m%d_%H%M%S_%f}"}}
            path = launch_dir / f"worker_{index:02d}.yaml"
            OmegaConf.save(OmegaConf.create(worker), path)
            # The suite can select a process-local I/O wrapper. Model/config
            # identities and all generation code stay unchanged for resume.
            module = "optics_experiments"
            if (os.environ.get("OPTICS_BENCHMARK_RUNTIME_WORKER") == "1"
                    and cfg["benchmark"]["architecture"] in {"qwen_draw_align", "direct_draw_align"}):
                module = "scripts.benchmark_runtime_worker"
            argv = [options.get("python") or sys.executable, "-u", "-m", module,
                    "--config-path", str(launch_dir), "--config-name", path.stem]
            item = {"shard_index": index, "argv": argv,
                    "log": str(launch_dir / f"worker_{index:02d}.log")}
            if not remote_api:
                item["cuda_visible_devices"] = device
            plan.append(item)
        report = {"status": "plan", "architecture": cfg["benchmark"]["architecture"],
                  "benchmark_count": len(sources), "selected_count": selected_count, "workers": plan}
        atomic_json(launch_dir / "launch.json", report)
        if options.get("dry_run", False) or not cfg["benchmark"].get("execute", True):
            return report
        if remote_api and options.get("preflight", False):
            from .benchmark_preflight import check_remote_endpoints
            check_remote_endpoints(cfg, logger=logger)
        if not remote_api:
            _check_actor(cfg)
        if logger:
            logger.event("benchmark_launch", workers=count, samples=selected_count, launch_dir=str(launch_dir),
                         worker_backend="remote_api" if remote_api else "local_gpu")
        for item in plan:
            environment = dict(os.environ)
            for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
                environment.pop(name, None)
            environment["PYTHONUNBUFFERED"] = "1"
            if not remote_api:
                environment["CUDA_VISIBLE_DEVICES"] = item["cuda_visible_devices"]
            environment.setdefault("OMP_NUM_THREADS", "4")
            environment.setdefault("TOKENIZERS_PARALLELISM", "false")
            if not remote_actor and cfg["benchmark"]["architecture"] in {"qwen_sensenova", "qwen_draw_align"}:
                environment.setdefault(cfg["benchmark"]["actor"].get("api_key_env", "OPTICS_TEACHER_API_KEY"), "local-benchmark")
            stream = open(item["log"], "w", encoding="utf-8")
            streams.append(stream)
            children.append(subprocess.Popen(item["argv"], env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
        started, last_update = time.monotonic(), 0.0
        previous_issue = None
        selected_ids = {row["sample_id"] for row in sources[:selected_count]}
        while True:
            active = sum(process.poll() is None for process in children)
            now = time.monotonic()
            if now - last_update >= max(1, float(options.get("progress_interval", 10))) or not active:
                progress = _progress(root, count, selected_ids, cache=progress_cache)
                completed, total = progress["completed"], progress["total"]
                filled = int(20 * completed / total) if total else 20
                elapsed = now - started
                reasons = Counter(progress["failure_stop_reasons"]).most_common(3)
                reason_text = " stop_reasons=" + ",".join(f"{reason}:{count}" for reason, count in reasons) if reasons else ""
                print(f"benchmark [{'=' * filled}{'-' * (20 - filled)}] completed={completed}/{total} "
                      f"pending={progress['pending']} submitted={progress['submitted']} images={progress['images']} "
                      f"budget_limited={progress['budget_limited']} technical_failed={progress['technical_failed']} "
                      f"not_submitted={progress['not_submitted']} active_workers={active} elapsed={elapsed:.0f}s{reason_text} "
                      f"stale_manifest_shards={progress['stale_manifest_shards']} "
                      f"unavailable_manifest_shards={progress['unavailable_manifest_shards']}", flush=True)
                issue = progress["latest_issue"]
                if issue and issue != previous_issue:
                    print(f"benchmark latest_issue sample={issue['sample_id']} status={issue['status']} "
                          f"reason={issue['stop_reason']} image_available={str(issue['image_available']).lower()} "
                          f"log={plan[issue['shard_index']]['log']} "
                          f"attempt={root / 'shards' / ('shard_%02d' % issue['shard_index']) / issue['attempt_directory']}",
                          flush=True)
                    previous_issue = issue
                if logger:
                    logger.event("benchmark_progress", **progress, active_workers=active, elapsed_seconds=elapsed)
                    logger.log({"benchmark/" + key: value for key, value in {
                        **progress, "active_workers": active, "elapsed_seconds": elapsed}.items()
                                if isinstance(value, (int, float))})
                last_update = now
            if not active:
                break
            time.sleep(1)
        result = merge(cfg, required_shards=set(progress_cache) | {
            index for index, process in enumerate(children) if process.returncode == 0})
        errors = [{"shard_index": i, "exit_code": p.returncode, "log": plan[i]["log"]}
                  for i, p in enumerate(children) if p.returncode]
        if errors:
            result.update(status="failed", exit_code=1, worker_errors=errors)
            atomic_json(root / "summary.json", result)
        return result
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        raise
    except Exception as error:
        coordinator_error = type(error).__name__
        raise
    finally:
        # Do not let a second termination signal interrupt child cleanup.
        for number in previous_handlers:
            signal.signal(number, signal.SIG_IGN)
        try:
            _stop(children)
            if (interrupted or coordinator_error) and children:
                # Even an unexpected coordinator error must preserve a usable
                # export when all required snapshots can be read and validated.
                try:
                    recovered = merge(cfg, required_shards=set(progress_cache) | {
                        index for index, process in enumerate(children) if process.returncode == 0})
                    recovered.update(status="interrupted" if interrupted else "failed",
                                     exit_code=130 if interrupted else 1,
                                     coordinator_error=coordinator_error,
                                     recovered_after_coordinator_stop=True)
                    atomic_json(root / "summary.json", recovered)
                    print(f"benchmark preserved export: images={recovered['images']} "
                          f"pending={recovered['pending']} summary={root / 'summary.json'}", flush=True)
                except Exception as error:
                    print(f"benchmark recovery export unavailable: {type(error).__name__}; "
                          f"existing images/attempts remain in {root / 'shards'}", flush=True)
        finally:
            for stream in streams:
                stream.close()
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)
