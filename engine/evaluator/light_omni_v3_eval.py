"""V3 generation with optional, independently resumable image-grounded judging.

The V1/V2 generation evaluator is deliberately left unchanged. VLM artifacts
live beside, never in place of, generated images and generation result JSONs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
from typing import Any
import uuid

from tqdm import tqdm

from engine.evaluator.light_omni_eval import LightOmniEvaluator, ResultConflictError


def summarize_answers(answers):
    """C contributes zero to the main score, not an excluded hard example."""
    counts = Counter(
        answer.get("choice") for answer in answers
        if answer.get("status") == "success" and isinstance(answer.get("choice"), str)
        and answer["choice"] in {"A", "B", "C"}
    )
    total = len(answers)
    answered = sum(counts.values())
    determinate = counts["A"] + counts["B"]
    return {
        "requested_questions": total,
        "answered_questions": answered,
        "technical_failures": total - answered,
        "passed_questions": counts["A"],
        "failed_questions": counts["B"],
        "cannot_determine_questions": counts["C"],
        "score": counts["A"] / answered if answered else None,
        "strict_pass_rate": counts["A"] / total if total else None,
        "determinate_accuracy": counts["A"] / determinate if determinate else None,
        "answer_coverage": answered / total if total else 0.0,
        "determinate_coverage": determinate / answered if answered else 0.0,
    }


class LightOmniV3Evaluator(LightOmniEvaluator):
    def __init__(
        self,
        compute_vlm_metric: bool = False,
        vlm_judge=None,
        vlm_max_workers: int = 4,
        vlm_force_recompute: bool = False,
        vlm_fail_on_error: bool = False,
        vlm_result_subdir: str = "vlm_results",
        metrics_only: bool = False,
        **generation_options: Any,
    ):
        super().__init__(**generation_options)
        self.compute_vlm_metric = bool(compute_vlm_metric)
        self.metrics_only = bool(metrics_only)
        self.requires_client = not self.metrics_only
        self.vlm_judge = vlm_judge
        self.vlm_max_workers = int(vlm_max_workers)
        self.vlm_force_recompute = bool(vlm_force_recompute)
        self.vlm_fail_on_error = bool(vlm_fail_on_error)
        self.vlm_result_subdir = vlm_result_subdir
        if self.vlm_max_workers < 1:
            raise ValueError("vlm_max_workers must be positive")
        if self.metrics_only and not self.compute_vlm_metric:
            raise ValueError("metrics_only requires compute_vlm_metric=true")
        if self.compute_vlm_metric:
            if self.vlm_judge is None:
                raise ValueError("VLM scoring requires evaluator.vlm_judge")
            if not self.save_images or not self.save_json:
                raise ValueError("VLM scoring requires save_images=true and save_json=true for safe resume")
        self._future_lock = threading.Lock()

    def _preflight(self, client, dataset, output_dir):
        reserved = [self.image_subdir, self.result_subdir, self.distributed_result_subdir,
                    self.vlm_result_subdir]
        targets = [(output_dir / path).resolve() for path in reserved]
        if any(not path.is_relative_to(output_dir) or path == output_dir for path in targets):
            raise ValueError("V3 artifact subdirectories must stay inside output_dir")
        if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
               for i, a in enumerate(targets) for b in targets[i + 1:]):
            raise ValueError("Generation and VLM artifact subdirectories must be separate")
        if self.metrics_only:
            if not (output_dir / self.image_subdir).is_dir() or not (output_dir / self.result_subdir).is_dir():
                raise ValueError("metrics_only requires an existing output directory containing images/ and results/")
            for sample in dataset:
                self._validate_saved_identity(sample, output_dir)
        else:
            super()._preflight(client, dataset, output_dir)
        if self.compute_vlm_metric:
            for sample in dataset:
                if not sample.get("questions"):
                    raise ValueError(f"{sample['sample_id']}: V3 scoring requires current questions")
        if not self.metrics_only:
            self._validate_lora_identity(client, output_dir)

    def _validate_lora_identity(self, client, output_dir):
        """Rank-zero preflight under the parent's output lock, separate from PNG IDs.

        Hydra generation fingerprints predate adapter-content identities. Keep
        those fingerprints stable, while refusing to reuse a LoRA output tree
        whose actual loaded adapter hashes differ or were never recorded.
        """
        settings = getattr(client, "cfg", None)
        identity = settings.get("lora_identity") if isinstance(settings, Mapping) else None
        if identity is None:
            return  # Existing non-LoRA generation and score-only jobs are unchanged.
        if not isinstance(identity, Mapping) or set(identity) != {"base_family", "adapter_sha256"}:
            raise ResultConflictError("Invalid client LoRA content identity")
        family, hashes = identity["base_family"], identity["adapter_sha256"]
        if not isinstance(family, str) or not family or not isinstance(hashes, Mapping) or not hashes:
            raise ResultConflictError("Invalid client LoRA content identity")
        if any(not isinstance(name, str) or not name or not isinstance(value, str)
               or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
               for name, value in hashes.items()):
            raise ResultConflictError("Invalid client LoRA SHA256 identity")
        expected = {"base_family": family, "adapter_sha256": dict(hashes)}
        path = output_dir / ".light_omni_lora_identity.json"
        if path.is_symlink():
            raise ResultConflictError("LoRA identity sidecar must not be a symlink")
        if path.exists():
            try:
                saved = self._load_json(path)
            except (OSError, ValueError) as exc:
                raise ResultConflictError(f"Unreadable LoRA identity sidecar: {path}") from exc
            if (not isinstance(saved, dict) or type(saved.get("format_version")) is not int
                    or saved["format_version"] != 1 or saved.get("lora_identity") != expected):
                raise ResultConflictError("LoRA adapter contents differ from the output directory identity; choose a new output directory")
            return
        # Check the entire tree, not just the selected sample subset. A missing
        # sidecar must never retroactively certify images from unknown weights.
        for subdir in (self.image_subdir, self.result_subdir, self.distributed_result_subdir):
            artifact_dir = output_dir / subdir
            if artifact_dir.exists() and any(path.is_file() or path.is_symlink() for path in artifact_dir.rglob("*")):
                raise ResultConflictError(
                    "Existing LoRA generation artifacts have no .light_omni_lora_identity.json; "
                    "cannot verify their adapter contents. Use a new output directory or explicitly "
                    "audit provenance before creating the sidecar. allow_legacy_resume does not bypass this check."
                )
        self._save_json(path, {"format_version": 1, "lora_identity": expected})

    def _validate_saved_identity(self, sample, output_dir):
        """Check prompt identity without loading, or pretending to know, a client."""
        saved = self._read_result(output_dir / self.result_subdir / f"{sample['sample_id']}.json")
        info = self._image_info(output_dir / self.image_subdir / f"{sample['sample_id']}.png", verify=False)
        if saved is None:
            return  # Record missing/unreadable generation metadata as a technical failure.
        if saved.get("sample_id") != sample["sample_id"] or saved.get("generation_prompt", saved.get("prompt_en")) != sample["generation_prompt"]:
            raise ResultConflictError(f"{sample['sample_id']}: saved prompt differs from V3; regenerate into a new output directory")
        if info is not None:
            saved_key = saved.get("generation_fingerprint")
            image_key = info.get("light_omni_generation")
            if saved_key and image_key and saved_key != image_key:
                raise ResultConflictError(f"{sample['sample_id']}: PNG and generation JSON fingerprints disagree")
            if (not saved_key or not image_key) and not self.allow_legacy_resume:
                raise ResultConflictError(f"{sample['sample_id']}: cannot verify legacy image; confirm provenance before allow_legacy_resume=true")

    def _one(self, index, sample, client, image_dir, result_dir, rank):
        # Only generation/storage errors use the generation retry policy.
        result = super()._one(index, sample, client, image_dir, result_dir, rank)
        if self.compute_vlm_metric:
            self._submit_score(sample, result, image_dir / f"{sample['sample_id']}.png", rank)
        return result

    def _submit_score(self, sample, generated, image_path, rank):
        future = self._score_pool.submit(self._score_sample, sample, generated, image_path, rank)
        with self._future_lock:
            self._score_futures.append(future)

    def _evaluate(self, client, dataset, output_dir, dist_ctx):
        if not self.compute_vlm_metric:
            return super()._evaluate(client, dataset, output_dir, dist_ctx)
        self._vlm_dir = output_dir / self.vlm_result_subdir
        self._judge_fingerprint = self.vlm_judge.settings_fingerprint()
        # Catch differing model aliases/protocols across torchrun workers.
        identity = self._digest({"judge": self._judge_fingerprint,
                                 "metrics_only": self.metrics_only,
                                 "vlm_result_subdir": self.vlm_result_subdir})
        if len(set(dist_ctx.all_gather(identity))) != 1:
            raise ResultConflictError("V3 ranks disagree on VLM model or scoring protocol")
        self._score_futures = []
        self._score_pool = ThreadPoolExecutor(max_workers=self.vlm_max_workers, thread_name_prefix="light-omni-vlm")
        generation_summary = None
        generation_results = []
        scored = []
        worker_error = None
        try:
            if self.metrics_only:
                for index in range(dist_ctx.rank, len(dataset), dist_ctx.world_size):
                    sample = dataset[index]
                    result_path = output_dir / self.result_subdir / f"{sample['sample_id']}.json"
                    generated = self._read_result(result_path)
                    if generated is None:
                        generated = {"status": "failed", "error": "Missing or unreadable generation result JSON"}
                    self._submit_score(sample, generated, output_dir / self.image_subdir / f"{sample['sample_id']}.png", dist_ctx.rank)
            else:
                # Judges run in a separate pool while GPU generation continues.
                # Keep one result collective for both stages. Calling the parent's
                # distributed loop here would let an early setup exception skip its
                # generation collective and accidentally match the scoring collective.
                generation_results = self._generate_local(client, dataset, output_dir, dist_ctx)
            for future in tqdm(as_completed(self._score_futures), total=len(self._score_futures),
                               desc=f"Judging Light-Omni V3 [rank {dist_ctx.rank}]", dynamic_ncols=True):
                scored.append(future.result())
        except Exception as exc:
            worker_error = f"rank {dist_ctx.rank}: {exc}"
        finally:
            for future in self._score_futures:
                future.cancel()
            self._score_pool.shutdown(wait=True, cancel_futures=True)

        payloads = dist_ctx.all_gather({"results": scored, "generation_results": generation_results,
                                       "error": worker_error})
        errors = [item["error"] for item in payloads if item["error"]]
        if errors:
            raise RuntimeError("Light-Omni V3 scoring worker failed: " + "; ".join(errors))
        all_scored = [row for payload in payloads for row in payload["results"]]
        total_failures = sum(row["technical_failures"] for row in all_scored)
        if dist_ctx.is_main_process:
            order = {dataset[i]["sample_id"]: i for i in range(len(dataset))}
            all_scored.sort(key=lambda row: order[row["sample_id"]])
            if not self.metrics_only:
                generated = [row for payload in payloads for row in payload["generation_results"]]
                generated.sort(key=lambda row: order[row["sample_id"]])
                generation_summary = self._save_generation_summary(generated, output_dir, dist_ctx)
            vlm_summary = self._aggregate(all_scored, dist_ctx)
            selected = set(order)
            known = {path.stem for path in self._vlm_dir.glob("*.json")}
            known.update(path.stem for path in (output_dir / self.result_subdir).glob("*.json"))
            suffix = ".partial." + self._digest(sorted(selected))[:12] if known - selected else ""
            vlm_summary["summary_scope"] = "selected_samples"
            vlm_summary["summary_file"] = f"vlm_summary{suffix}.json"
            vlm_summary["sample_ids"] = [row["sample_id"] for row in all_scored]
            self._save_json(output_dir / f"vlm_all_results{suffix}.json", all_scored)
            self._save_json(output_dir / vlm_summary["summary_file"], vlm_summary)
            if generation_summary is not None:
                generation_summary.update(scoring_enabled=True, vlm=vlm_summary)
                self._save_json(output_dir / generation_summary["summary_file"], generation_summary)
            print("======== Light-Omni V3 VLM Summary ========")
            print(json.dumps(vlm_summary, ensure_ascii=False, indent=2))
            summary = generation_summary if generation_summary is not None else vlm_summary
        else:
            summary = {"status": "worker_finished", "rank": dist_ctx.rank}
        # Fail only after preserving the partial scores. Never regenerate on a judge error.
        if self.vlm_fail_on_error and total_failures:
            raise RuntimeError(f"VLM scoring recorded {total_failures} technical failures; saved answers can be resumed")
        return summary

    def _generate_local(self, client, dataset, output_dir, dist_ctx):
        """No collectives: all setup and generation errors reach the same gather."""
        image_dir = output_dir / self.image_subdir
        result_dir = output_dir / self.result_subdir
        image_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        workers = self.max_workers
        if workers is None:
            workers = self.remote_api_workers if getattr(client, "model_type", None) == "remote-api" else 1
        indices = list(range(dist_ctx.rank, len(dataset), dist_ctx.world_size))
        print(json.dumps({"benchmark": "Light-Omni V3", "task": "generation + VLM",
                          "global_samples": len(dataset), "local_samples": len(indices),
                          "rank": dist_ctx.rank, "world_size": dist_ctx.world_size,
                          "generation_workers": workers, "vlm_workers": self.vlm_max_workers,
                          "output_dir": str(output_dir)}, indent=2))
        results = []
        with tqdm(total=len(indices), desc=f"Generating Light-Omni V3 [rank {dist_ctx.rank}]", dynamic_ncols=True) as progress:
            if workers == 1:
                for index in indices:
                    results.append(self._one(index, dataset[index], client, image_dir, result_dir, dist_ctx.rank))
                    progress.update(1)
            else:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="light-omni-gen") as pool:
                    futures = [pool.submit(self._one, index, dataset[index], client, image_dir, result_dir, dist_ctx.rank)
                               for index in indices]
                    try:
                        for future in as_completed(futures):
                            results.append(future.result())
                            progress.update(1)
                    finally:
                        for future in futures:
                            future.cancel()
        return results

    def _save_generation_summary(self, results, output_dir, dist_ctx):
        selected = {row["sample_id"] for row in results}
        other = {path.stem for path in (output_dir / self.result_subdir).glob("*.json")} - selected
        suffix = ".partial." + self._digest(sorted(selected))[:12] if other else ""
        categories = {}
        for row in results:
            counts = categories.setdefault(row["category_en"], {"total": 0, "success": 0, "failed": 0})
            counts["total"] += 1
            counts[row["status"]] += 1
        success = sum(row["status"] == "success" for row in results)
        summary = {
            "benchmark": "Light-Omni V3 revised", "task": "text-to-image generation + VLM scoring",
            "model_name": self._model_name, "client_fingerprint": self._client_fingerprint,
            "sample_ids": [row["sample_id"] for row in results],
            "total": len(results), "success": success, "failed": len(results) - success,
            "success_rate": success / len(results) if results else 0.0,
            "by_category_en": dict(sorted(categories.items())),
            "failed_sample_ids": [row["sample_id"] for row in results if row["status"] != "success"],
            "legacy_resumed_samples": sum(bool(row.get("legacy_resume_unverified")) for row in results),
            "distributed": dist_ctx.enabled, "world_size": dist_ctx.world_size,
            "summary_scope": "selected_samples", "other_saved_samples": len(other),
            "summary_file": f"summary{suffix}.json",
        }
        self._save_json(output_dir / f"all_results{suffix}.json", results)
        if dist_ctx.enabled:
            for rank in range(dist_ctx.world_size):
                self._save_json(output_dir / self.distributed_result_subdir / f"rank_{rank:05d}.json",
                                [row for row in results if row["rank"] == rank])
        return summary

    @staticmethod
    def _image_digest(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _valid_answer(value, question_id):
        return (isinstance(value, dict) and value.get("question_id") == question_id
                and value.get("status") == "success" and isinstance(value.get("choice"), str)
                and value["choice"] in {"A", "B", "C"}
                and isinstance(value.get("evidence"), list) and 1 <= len(value["evidence"]) <= 3
                and all(isinstance(line, str) and line.strip() for line in value["evidence"]))

    def _score_sample(self, sample, generated, image_path, rank):
        sample_id = sample["sample_id"]
        failure = None
        image_sha256 = None
        if generated.get("status") != "success":
            failure = "Generation did not succeed or its metadata is unavailable"
        elif self._image_info(image_path) is None:
            failure = "Generated image is missing, unreadable, or corrupt"
        else:
            image_sha256 = self._image_digest(image_path)
        answers = []
        for question in sample["questions"]:
            qid = question["id"]
            if failure:
                answers.append({"question_id": qid, "status": "technical_failure", "choice": None,
                                "evidence": [], "error": failure, "error_type": "generation_artifact", "attempts": 0})
                continue
            key = self._digest({
                "image_sha256": image_sha256,
                "generation_fingerprint": generated.get("generation_fingerprint"),
                "prompt": sample["generation_prompt"], "judge": self._judge_fingerprint,
                "question": self.vlm_judge.judge_question_payload(question),
            })
            cache_path = self._vlm_dir / sample_id / f"{key}.json"
            cached = None
            if cache_path.is_file():
                try:
                    cached = self._load_json(cache_path)
                except (OSError, ValueError):
                    pass
            if (not self.vlm_force_recompute and isinstance(cached, dict)
                    and cached.get("cache_fingerprint") == key and self._valid_answer(cached, qid)):
                answer = {**cached, "resumed": True}
            else:
                try:
                    answer = self.vlm_judge.judge_one(image_path, question, sample["generation_prompt"])
                    if not self._valid_answer(answer, qid) and (
                        not isinstance(answer, dict) or answer.get("status") != "technical_failure"
                    ):
                        raise ValueError("Judge returned an invalid answer object")
                except Exception as exc:
                    # Do not serialize arbitrary exception messages: transport exceptions
                    # can contain the gateway URL, including its secret ak query parameter.
                    answer = {"question_id": qid, "status": "technical_failure", "choice": None,
                              "evidence": [], "error": "Judge raised an unexpected exception",
                              "error_type": type(exc).__name__, "attempts": 0}
                answer = {**answer, "sample_id": sample_id, "cache_fingerprint": key,
                          "image_sha256": image_sha256, "judge_fingerprint": self._judge_fingerprint,
                          "resumed": False}
                if self.vlm_force_recompute and cached is not None:
                    self._save_json(cache_path.with_name(f"{key}.previous.{uuid.uuid4().hex}.json"), cached)
                self._save_json(cache_path, answer)
            answers.append(answer)
        if image_sha256 is not None and self._image_digest(image_path) != image_sha256:
            raise ResultConflictError(f"{sample_id}: image changed during VLM scoring")
        counts = summarize_answers(answers)
        result = {
            "sample_id": sample_id, "source_index": sample["source_index"], "rank": rank,
            "category_id": sample.get("category_id"), "category_en": sample["category_en"],
            "category_zh": sample["category_zh"], "benchmark_version": sample.get("benchmark_version"),
            "model_name": generated.get("model_name"), "generation_prompt": sample["generation_prompt"],
            "generation_fingerprint": generated.get("generation_fingerprint"),
            "generated_image_path": str(image_path), "image_sha256": image_sha256,
            "judge_fingerprint": self._judge_fingerprint, "judge": self.vlm_judge.safe_settings(),
            "question_bank_fingerprint": sample.get("question_bank_fingerprint"),
            "status": "success" if counts["technical_failures"] == 0 else "partial",
            **counts, "answers": answers,
        }
        self._save_json(self._vlm_dir / f"{sample_id}.json", result)
        return result

    def _aggregate(self, results, dist_ctx):
        answers = [answer for result in results for answer in result["answers"]]
        categories = {}
        for result in results:
            key = result.get("category_id") or result["category_en"]
            categories.setdefault(key, []).append(result)
        by_category = {}
        for key, rows in sorted(categories.items()):
            scores = [row["score"] for row in rows if row["score"] is not None]
            by_category[key] = {
                "category_en": rows[0]["category_en"], "category_zh": rows[0]["category_zh"],
                "samples": len(rows),
                **summarize_answers([answer for row in rows for answer in row["answers"]]),
                "macro_sample_score": sum(scores) / len(scores) if scores else None,
            }
        counts = summarize_answers(answers)
        scores = [row["score"] for row in results if row["score"] is not None]
        return {
            "benchmark": "Light-Omni V3 revised", "scoring_enabled": True,
            "benchmark_versions": sorted({str(row.get("benchmark_version")) for row in results}),
            "status": "complete" if counts["technical_failures"] == 0 else "partial",
            "model_names": sorted({row["model_name"] for row in results if row.get("model_name")}),
            "judge": self.vlm_judge.safe_settings(), "judge_fingerprint": self._judge_fingerprint,
            "questions_per_call": 1, "requested_samples": len(results),
            "fully_evaluated_samples": sum(row["technical_failures"] == 0 for row in results),
            **counts, "macro_sample_score": sum(scores) / len(scores) if scores else None,
            "resumed_questions": sum(bool(answer.get("resumed")) for answer in answers),
            "score_definition": "A/(A+B+C); C scores zero. Technical failures are separate; inspect answer_coverage.",
            "strict_pass_rate_definition": "A/requested_questions, including technically unanswered questions in denominator",
            "by_category": by_category, "vlm_workers_per_process": self.vlm_max_workers,
            "distributed": dist_ctx.enabled, "world_size": dist_ctx.world_size,
        }
