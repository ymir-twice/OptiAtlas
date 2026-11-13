"""Generation-only evaluator for the Light-Omni T2I prompt set."""

from __future__ import annotations

import inspect
import hashlib
import io
import logging
import math
import fcntl
import json
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from omegaconf import OmegaConf
import requests
from tqdm import tqdm

from engine.evaluator.base_eval import RateLimiter
from engine.utils.img_covt import b64_to_pil
from engine.utils.light_omni_runtime import LightOmniDistributedContext


logger = logging.getLogger(__name__)


class ResultConflictError(RuntimeError):
    """An existing artifact cannot be safely reused by this generation job."""


class LightOmniEvaluator:
    """Call ``generate(prompt_en)`` and save outputs without scoring them."""

    def __init__(
        self,
        save_images: bool = True,
        save_json: bool = True,
        image_subdir: str = "images",
        result_subdir: str = "results",
        distributed_result_subdir: str = "distributed_results",
        resume: bool = True,
        skip_existing_success: bool = True,
        retry_failed_existing: bool = True,
        max_retries: int = 3,
        retry_sleep_seconds: float = 5.0,
        retry_backoff: float = 2.0,
        requests_per_minute: Optional[float] = None,
        max_workers: Optional[int] = None,
        remote_api_workers: int = 4,
        fail_fast: bool = False,
        save_failed_json: bool = True,
        overwrite_images: bool = False,
        distributed: Optional[bool] = None,
        distributed_backend: Optional[str] = None,
        distributed_timeout_seconds: float = 1800.0,
        allow_legacy_resume: bool = False,
    ):
        if max_workers is not None and int(max_workers) < 1:
            raise ValueError("max_workers must be positive or null")
        if int(remote_api_workers) < 1:
            raise ValueError("remote_api_workers must be positive")
        self.save_images = bool(save_images)
        self.save_json = bool(save_json)
        self.image_subdir = image_subdir
        self.result_subdir = result_subdir
        self.distributed_result_subdir = distributed_result_subdir
        self.resume = bool(resume)
        self.skip_existing_success = bool(skip_existing_success)
        self.retry_failed_existing = bool(retry_failed_existing)
        self.max_retries = max(1, int(max_retries))
        self.retry_sleep_seconds = float(retry_sleep_seconds)
        self.retry_backoff = float(retry_backoff)
        self.requests_per_minute = requests_per_minute
        self.max_workers = None if max_workers is None else int(max_workers)
        self.remote_api_workers = int(remote_api_workers)
        self.fail_fast = bool(fail_fast)
        self.save_failed_json = bool(save_failed_json)
        self.overwrite_images = bool(overwrite_images)
        self.distributed = distributed
        self.distributed_backend = distributed_backend
        self.distributed_timeout_seconds = float(distributed_timeout_seconds)
        self.allow_legacy_resume = bool(allow_legacy_resume)
        for name, value, allow_zero in (
            ("retry_sleep_seconds", self.retry_sleep_seconds, True),
            ("retry_backoff", self.retry_backoff, False),
            ("distributed_timeout_seconds", self.distributed_timeout_seconds, False),
        ):
            if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                raise ValueError(f"Invalid {name}: {value}")
        if requests_per_minute is not None and (
            not math.isfinite(float(requests_per_minute)) or float(requests_per_minute) <= 0
        ):
            raise ValueError("requests_per_minute must be positive or null")
        self.rate_limiter = RateLimiter(requests_per_minute)
        self._rate_lock = threading.Lock()
        self._save_lock = threading.Lock()

    def evaluate(self, client, dataset, output_dir: Path, cfg=None) -> Dict[str, Any]:
        self._client_fingerprint = self._fingerprint_client(client, cfg)
        self._model_name = getattr(client, "model_name", type(client).__name__)
        dist_ctx = LightOmniDistributedContext(
            enabled=self.distributed, backend=self.distributed_backend,
            timeout_seconds=self.distributed_timeout_seconds,
        )
        output_lock = None
        try:
            output_dir = Path(output_dir).resolve()
            # All ranks must work on exactly the same samples and output tree.
            identity = self._digest({
                "output_dir": str(output_dir),
                "client": self._client_fingerprint,
                "samples": [dict(dataset[i]) for i in range(len(dataset))],
                "storage": [self.save_images, self.save_json, self.image_subdir, self.result_subdir],
            })
            if len(set(dist_ctx.all_gather(identity))) != 1:
                raise ResultConflictError("Light-Omni ranks disagree on dataset, model or output directory.")
            error = None
            if dist_ctx.is_main_process:
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_lock = (output_dir / ".light_omni.lock").open("a")
                    try:
                        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise ResultConflictError(f"Output directory is already in use by another Light-Omni job: {output_dir}") from exc
                    self._preflight(client, dataset, output_dir)
                except Exception as exc:
                    error = str(exc)
            errors = [e for e in dist_ctx.all_gather(error) if e]
            if errors:
                raise ResultConflictError("Light-Omni preflight failed: " + errors[0])
            return self._evaluate(client, dataset, output_dir, dist_ctx)
        finally:
            if output_lock is not None:
                output_lock.close()
            dist_ctx.close()

    def _evaluate(self, client, dataset, output_dir: Path, dist_ctx) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = output_dir / self.image_subdir
        result_dir = output_dir / self.result_subdir
        rank_dir = output_dir / self.distributed_result_subdir
        if self.save_images:
            image_dir.mkdir(parents=True, exist_ok=True)
        if self.save_json:
            result_dir.mkdir(parents=True, exist_ok=True)
            if dist_ctx.enabled:
                rank_dir.mkdir(parents=True, exist_ok=True)

        workers = self.max_workers
        if workers is None:
            workers = self.remote_api_workers if getattr(client, "model_type", None) == "remote-api" else 1
        indices = list(range(dist_ctx.rank, len(dataset), dist_ctx.world_size))
        print(json.dumps({
            "benchmark": "Light-Omni", "task": "text-to-image generation only",
            "global_samples": len(dataset), "local_samples": len(indices),
            "rank": dist_ctx.rank, "world_size": dist_ctx.world_size,
            "workers": workers, "output_dir": str(output_dir),
        }, indent=2, ensure_ascii=False))

        results = []
        worker_error = None
        progress = tqdm(total=len(indices), desc="Generating Light-Omni", dynamic_ncols=True)
        try:
            if workers == 1:
                for index in indices:
                    results.append(self._one(index, dataset[index], client, image_dir, result_dir, dist_ctx.rank))
                    progress.update(1)
            else:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="light-omni") as pool:
                    futures = {
                        pool.submit(self._one, i, dataset[i], client, image_dir, result_dir, dist_ctx.rank): i
                        for i in indices
                    }
                    try:
                        for future in as_completed(futures):
                            results.append(future.result())
                            progress.update(1)
                    except Exception:
                        # Context-manager shutdown otherwise runs every queued
                        # API request, even after fail_fast/storage failure.
                        for pending in futures:
                            pending.cancel()
                        raise
        except Exception as exc:
            worker_error = f"rank {dist_ctx.rank}: {exc}"
        finally:
            progress.close()
        results.sort(key=lambda item: item["index"])

        payloads = dist_ctx.all_gather({"results": results, "error": worker_error})
        errors = [payload["error"] for payload in payloads if payload["error"]]
        if errors:
            raise RuntimeError("Light-Omni worker failed: " + "; ".join(errors))
        if dist_ctx.enabled:
            if self.save_json:
                self._save_json(rank_dir / f"rank_{dist_ctx.rank:05d}.json", results)
            if not dist_ctx.is_main_process:
                return {"status": "worker_finished", "rank": dist_ctx.rank}
            results = [item for payload in payloads for item in payload["results"]]
            results.sort(key=lambda item: item["index"])

        successful = [item for item in results if item.get("status") == "success"]
        categories: Dict[str, Dict[str, int]] = {}
        for item in results:
            category = item.get("category_en", "unknown")
            entry = categories.setdefault(category, {"total": 0, "success": 0, "failed": 0})
            entry["total"] += 1
            entry[item.get("status", "failed")] += 1
        summary = {
            "benchmark": "Light-Omni", "task": "text-to-image generation only",
            "model_name": self._model_name,
            "client_fingerprint": self._client_fingerprint,
            "sample_ids": [item["sample_id"] for item in results],
            "scoring_enabled": False, "total": len(results),
            "success": len(successful), "failed": len(results) - len(successful),
            "success_rate": len(successful) / len(results) if results else 0.0,
            "by_category_en": dict(sorted(categories.items())),
            "failed_sample_ids": [item["sample_id"] for item in results if item.get("status") != "success"],
            "thread_workers_per_process": workers,
            "legacy_resumed_samples": sum(bool(item.get("legacy_resume_unverified")) for item in results),
            "distributed": dist_ctx.enabled, "world_size": dist_ctx.world_size,
        }
        if self.save_json:
            selected_ids = {item["sample_id"] for item in results}
            other_ids = {path.stem for path in result_dir.glob("*.json")} - selected_ids
            # A two-sample smoke test in a populated directory must not replace
            # the full run's aggregate with a two-sample summary.
            suffix = ".partial." + self._digest(sorted(selected_ids))[:12] if other_ids else ""
            summary["summary_scope"] = "selected_samples"
            summary["other_saved_samples"] = len(other_ids)
            summary["summary_file"] = f"summary{suffix}.json"
            self._save_json(output_dir / f"all_results{suffix}.json", results)
            self._save_json(output_dir / summary["summary_file"], summary)
        print("======== Light-Omni Summary ========")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    @staticmethod
    def _digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    @classmethod
    def _fingerprint_client(cls, client, cfg):
        # Fingerprint inference settings, but never persist credentials. Device
        # routing/offload and endpoint authentication do not define a generation.
        excluded = {
            "device", "auto_local_rank", "offload_folder", "max_mem_per_gpu",
            "model_cpu_offload", "sequential_cpu_offload", "cpu_offload", "vram_mode",
            "load_on_init", "disable_progress_bar", "base_url", "endpoint_path",
            "log_id", "timeout_seconds", "download_timeout_seconds",
        }

        def clean(value):
            if OmegaConf.is_config(value):
                # Resolve only retained fields, not unused credentials.
                value = OmegaConf.to_container(value, resolve=False)
            if isinstance(value, Mapping):
                return {
                    str(k): clean(v) for k, v in value.items()
                    if str(k) not in excluded
                    and not any(word in str(k).lower() for word in ("api_key", "password", "secret", "credential", "access_token"))
                }
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            if value is None or isinstance(value, (str, bool, int, float)):
                return value
            return str(value)

        if cfg is not None and cfg.get("client") is not None:
            settings = clean(cfg["client"])
            if OmegaConf.is_config(cfg):
                # Copy retained values from the original config to support env
                # interpolations in model paths, seeds, etc.
                for key in settings:
                    settings[key] = clean(cfg.client[key])
        else:
            settings = clean(getattr(client, "cfg", {}))
            for key in (
                "model_name", "model_type", "model_path", "width", "height", "seed",
                "dtype", "num_inference_steps", "guidance_scale", "true_cfg_scale",
                "negative_prompt", "max_sequence_length", "image_quality",
            ):
                if hasattr(client, key):
                    settings[key] = clean(getattr(client, key))
        return cls._digest({"class": type(client).__module__ + "." + type(client).__qualname__, "config": settings})

    def _generation_key(self, sample):
        return self._digest({"client": self._client_fingerprint, "prompt": sample.get("generation_prompt", sample["prompt_en"])})

    @staticmethod
    def _read_result(path):
        if not path.is_file():
            return None
        try:
            value = LightOmniEvaluator._load_json(path)
            if not isinstance(value, dict) or value.get("status") not in {"success", "failed"}:
                raise ValueError("invalid result object")
            return value
        except (ValueError, OSError) as error:
            logger.warning("Unreadable Light-Omni result %s: %s", path, error)
            return None

    @staticmethod
    def _image_info(path, verify=True):
        if not path.is_file():
            return None
        try:
            with Image.open(path) as image:
                if verify:
                    image.load()  # Existence alone does not detect corrupt PNGs.
                return dict(image.info)
        except (OSError, ValueError):
            return None

    def _check_identity(self, sample, existing, image_info):
        expected = self._generation_key(sample)
        if existing:
            actual_prompt = existing.get("generation_prompt", existing.get("prompt_en"))
            if existing.get("sample_id") != sample["sample_id"] or actual_prompt != sample.get("generation_prompt", sample["prompt_en"]):
                raise ResultConflictError(f"{sample['sample_id']}: saved prompt/sample identity differs; choose a new output directory.")
            saved_key = existing.get("generation_fingerprint")
            if saved_key is not None and saved_key != expected:
                raise ResultConflictError(f"{sample['sample_id']}: model or generation settings changed; choose a new output directory.")
        if image_info is not None and image_info.get("light_omni_generation") not in (None, expected):
            raise ResultConflictError(f"{sample['sample_id']}: PNG belongs to a different generation.")
        legacy = (
            (existing is not None and not existing.get("generation_fingerprint"))
            or (image_info is not None and not image_info.get("light_omni_generation"))
        )
        if legacy and not self.allow_legacy_resume:
            raise ResultConflictError(
                f"{sample['sample_id']}: legacy output has no verifiable model/config fingerprint. "
                "Use a new output directory, or set evaluator.allow_legacy_resume=true "
                "only after confirming these outputs used the same model and settings."
            )
        return legacy

    def _preflight(self, client, dataset, output_dir):
        if getattr(client, "model_type", None) != "remote-api" and (self.max_workers or 1) > 1:
            raise ValueError("Local model clients are not thread-safe; use evaluator.max_workers=1 and torchrun for parallelism.")
        if not callable(getattr(client, "generate", None)):
            raise TypeError("Light-Omni requires client.generate(prompt).")
        for subdir in (self.image_subdir, self.result_subdir, self.distributed_result_subdir):
            target = (output_dir / subdir).resolve()
            if not target.is_relative_to(output_dir) or target == output_dir:
                raise ValueError("Light-Omni subdirectories must be inside output_dir.")
        legacy_count = 0
        for sample in dataset:
            image_path = output_dir / self.image_subdir / f"{sample['sample_id']}.png"
            result_path = output_dir / self.result_subdir / f"{sample['sample_id']}.json"
            existing = self._read_result(result_path)
            # Header metadata is enough for identity preflight. Each worker
            # fully decodes only its own samples before counting them as done.
            info = self._image_info(image_path, verify=False) if self.save_images else None
            if self.resume and existing and existing["status"] == "failed" and not self.retry_failed_existing:
                legacy_count += self._check_identity(sample, existing, info)
                continue
            # A deliberate rerun requires explicit image replacement when a
            # valid PNG already exists. Never call the model and discard it.
            if (not self.resume or not self.skip_existing_success):
                if self.save_images and image_path.exists() and not self.overwrite_images:
                    raise ResultConflictError(f"{image_path} exists; use a new output directory or enable overwrite_images for an explicit rerun.")
                continue
            legacy_count += self._check_identity(sample, existing, info)
        if legacy_count:
            logger.warning("Resuming %d legacy artifacts: their original model/settings cannot be verified.", legacy_count)

    def _base_result(self, index, sample, rank):
        return {
            **dict(sample), "index": index, "rank": rank,
            "generation_prompt": sample.get("generation_prompt", sample["prompt_en"]),
            "model_name": self._model_name, "client_fingerprint": self._client_fingerprint,
            "generation_fingerprint": self._generation_key(sample),
            "status": "failed", "attempt_num": 0, "generated_image_path": None,
            "metrics": {}, "error": None, "traceback": None,
        }

    def _one(self, index, sample: Mapping[str, Any], client, image_dir: Path, result_dir: Path, rank: int):
        sample_id = str(sample["sample_id"])
        result_path = result_dir / f"{sample_id}.json"
        image_path = image_dir / f"{sample_id}.png"
        existing = self._read_result(result_path)
        info = self._image_info(image_path) if self.save_images else None
        legacy = False
        if self.resume and existing and existing["status"] == "failed" and not self.retry_failed_existing:
            self._check_identity(sample, existing, info)
            return {**existing, **dict(sample), "index": index, "rank": rank}
        if self.resume and self.skip_existing_success:
            legacy = self._check_identity(sample, existing, info)
            if (self.save_images and info is not None) or (
                not self.save_images and existing and existing["status"] == "success"
            ):
                result = {**(existing or {}), **self._base_result(index, sample, rank)}
                result.update(
                    status="success", attempt_num=existing.get("attempt_num", 0) if existing else 0,
                    generated_image_path=(str(image_path) if self.save_images else existing.get("generated_image_path")),
                    resumed=True, resumed_from_generated_image=not existing or existing["status"] != "success",
                    legacy_resume_unverified=legacy,
                )
                # Never turn an unverifiable legacy artifact into a verified one.
                if legacy:
                    result["model_name"] = existing.get("model_name") if existing else None
                    result["client_fingerprint"] = None
                    result["generation_fingerprint"] = None
                if self.save_json:
                    self._save_json(result_path, result)
                return result

        if self.save_images and image_path.exists() and info is None:
            # Preserve corrupt artifacts for inspection, then regenerate.
            image_path.rename(image_path.with_name(image_path.name + f".corrupt.{uuid.uuid4().hex}"))
        result = self._base_result(index, sample, rank)
        image = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # RateLimiter stores mutable request timing state.
                with self._rate_lock:
                    self.rate_limiter.wait()
                response = self._call_generate(client, result["generation_prompt"])
                image = self._to_pil(response)
                result.update(status="success", attempt_num=attempt, error=None, traceback=None)
                break
            except Exception as error:
                result.update(status="failed", attempt_num=attempt, error=repr(error), traceback=traceback.format_exc())
                logger.error("[Rank %d] %s attempt %d/%d: %s", rank, sample_id, attempt, self.max_retries, error)
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep_seconds * self.retry_backoff ** (attempt - 1))
        # Storage errors must not trigger another paid/API or GPU generation.
        if result["status"] == "success":
            if self.save_images:
                self._save_image(image_path, image, result["generation_fingerprint"])
                result["generated_image_path"] = str(image_path)
            if self.save_json:
                self._save_json(result_path, result)
        else:
            if self.save_json and self.save_failed_json:
                self._save_json(result_path, result)
            if self.fail_fast:
                raise RuntimeError(result["error"])
        return result

    @staticmethod
    def _call_generate(client, prompt: str):
        method = client.generate
        try:
            inspect.signature(method).bind(prompt=prompt)
        except (TypeError, ValueError):
            return method(prompt)
        return method(prompt=prompt)

    @classmethod
    def _to_pil(cls, response) -> Image.Image:
        if isinstance(response, Image.Image):
            return response.convert("RGB")
        if isinstance(response, str):
            if response.startswith(("https://", "http://")):
                # Some OpenAI-compatible gateways return URL-only image data.
                with requests.get(response, timeout=120) as downloaded:
                    downloaded.raise_for_status()
                    with Image.open(io.BytesIO(downloaded.content)) as image:
                        return image.convert("RGB")
            return b64_to_pil(response).convert("RGB")
        if isinstance(response, (list, tuple)):
            if len(response) != 1:
                raise ValueError(f"Light-Omni expects one image per prompt, got {len(response)}; set n/batch_size=1.")
            return cls._to_pil(response[0])
        if isinstance(response, dict):
            if isinstance(response.get("image"), Image.Image):
                return response["image"].convert("RGB")
            if response.get("b64_json"):
                return b64_to_pil(response["b64_json"]).convert("RGB")
            if response.get("data"):
                return cls._to_pil(response["data"])
            if response.get("url"):
                return cls._to_pil(response["url"])
        if hasattr(response, "data"):
            return cls._to_pil(response.data)
        if getattr(response, "b64_json", None):
            return b64_to_pil(response.b64_json).convert("RGB")
        if isinstance(getattr(response, "image", None), Image.Image):
            return response.image.convert("RGB")
        if getattr(response, "url", None):
            return cls._to_pil(response.url)
        raise TypeError(f"Unsupported generation response type: {type(response).__name__}")

    def _save_image(self, path: Path, image: Image.Image, generation_key: str) -> None:
        with self._save_lock:
            if path.exists() and not self.overwrite_images:
                raise ResultConflictError(f"Refusing to silently discard a generated image: {path} exists.")
            temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
            metadata = PngInfo()
            metadata.add_text("light_omni_generation", generation_key)
            image.convert("RGB").save(temporary, format="PNG", pnginfo=metadata)
            os.replace(temporary, path)

    def _save_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        if path.is_file():
            try:
                self._load_json(path)
            except (ValueError, OSError):
                path.rename(path.with_name(path.name + f".corrupt.{uuid.uuid4().hex}"))
        os.replace(temporary, path)

    @staticmethod
    def _load_json(path: Path):
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
