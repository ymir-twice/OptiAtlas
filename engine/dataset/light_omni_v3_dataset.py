"""Current Light-Omni V3 prompts and image-grounded, judge-facing questions."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from engine.dataset.light_omni_dataset import LightOmniDataset


class LightOmniV3Dataset(LightOmniDataset):
    """Preserve source identity and expose no ideal answers or provenance.

    Slicing uses retained-record positions. ``source_index`` is the original
    benchmark index, which deliberately has gaps in the candidate release.
    Questions are a strict English allowlist suitable for one-question calls.
    """

    QUESTION_FIELDS = ("id", "question", "check", "rubric", "options", "judge_hint")
    OPTION_TEXTS = {"A": "True", "B": "False", "C": "Cannot determine"}

    def __init__(
        self,
        metadata_file: str | Path = "data/light_omni_v3_revised/light_omni_v3_revised.json",
        start_index: int = 0,
        max_samples: Optional[int] = None,
    ):
        super().__init__(metadata_file, start_index=start_index, max_samples=max_samples)

    @staticmethod
    def _digest(value: Any) -> str:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _text(value: Any, location: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{location} must be a non-empty string")
        return value

    @classmethod
    def _preserved_id(cls, value: Any, location: str) -> str:
        value = cls._text(value, location)
        if value != value.strip() or cls._filename_sample_id(value) != value:
            raise ValueError(f"{location} must be a safe filename ID; V3 source IDs cannot be rewritten")
        return value

    @classmethod
    def _question(cls, question: Any, location: str) -> Dict[str, Any]:
        if not isinstance(question, dict):
            raise TypeError(f"{location} must be a JSON object")
        qid = cls._preserved_id(question.get("id"), f"{location}.id")
        if question.get("type") != "tf_unknown":
            raise ValueError(f"{location}.type must be tf_unknown for A/B/C judging")
        if question.get("answer") != "A":
            raise ValueError(f"{location}.answer must be A (the ideal passing condition)")
        if question.get("target_missing_answer", "B") != "B":
            raise ValueError(f"{location}.target_missing_answer must be B")

        # Copy only the current English fields, never answer, translation or
        # historical human/model choices. Even nested option extras are dropped.
        result = {"id": qid}
        for field in ("question", "check"):
            result[field] = cls._text(question.get(field), f"{location}.{field}")
        rubric = question.get("rubric")
        if not isinstance(rubric, dict) or set(rubric) != set(cls.OPTION_TEXTS):
            raise ValueError(f"{location}.rubric must contain exactly A, B and C")
        result["rubric"] = {
            key: cls._text(rubric[key], f"{location}.rubric.{key}")
            for key in cls.OPTION_TEXTS
        }
        options = question.get("options")
        if not isinstance(options, list) or len(options) != 3:
            raise ValueError(f"{location}.options must contain exactly three A/B/C options")
        normalized_options = {}
        for option in options:
            if not isinstance(option, dict):
                raise TypeError(f"{location}.options must contain JSON objects")
            key = option.get("key")
            if not isinstance(key, str) or key not in cls.OPTION_TEXTS or key in normalized_options:
                raise ValueError(f"{location}.options must have distinct A, B and C keys")
            if option.get("text") != cls.OPTION_TEXTS[key]:
                raise ValueError(f"{location}.options.{key} must mean {cls.OPTION_TEXTS[key]!r}")
            normalized_options[key] = {"key": key, "text": option["text"]}
        result["options"] = [normalized_options[key] for key in cls.OPTION_TEXTS]
        hint = question.get("judge_hint", "")
        if not isinstance(hint, str):
            raise ValueError(f"{location}.judge_hint must be a string if present")
        result["judge_hint"] = hint
        for field in ("question", "check", "rubric", "judge_hint"):
            alias = field + "_en"
            if alias in question and question[alias] != result[field]:
                raise ValueError(f"{location}: current {field} disagrees with its English alias")
        return result

    def _load(self) -> list[Dict[str, Any]]:
        raw_records = self._read_records()
        self.metadata_sha256 = hashlib.sha256(self.metadata_file.read_bytes()).hexdigest()
        records = []
        seen_sample_ids, seen_source_indices, seen_question_ids = set(), set(), set()
        for record_index, record in enumerate(raw_records):
            location = f"Light-Omni V3 record {record_index}"
            sample_id = self._preserved_id(record.get("sample_id"), f"{location}.sample_id")
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id: {sample_id!r}")
            seen_sample_ids.add(sample_id)
            source_index = record.get("index")
            if type(source_index) is not int or source_index < 0:
                raise ValueError(f"{location}.index must be a non-negative source integer")
            if source_index in seen_source_indices:
                raise ValueError(f"Duplicate source index: {source_index}")
            seen_source_indices.add(source_index)
            sample = {"sample_id": sample_id, "source_index": source_index}
            for field in (
                "category_id", "category_en", "category_zh", "prompt_en",
                "prompt_zh", "generation_prompt", "benchmark_version",
            ):
                sample[field] = self._text(record.get(field), f"{location}.{field}")
            if "domain_zh" in record:
                sample["domain_zh"] = self._text(record["domain_zh"], f"{location}.domain_zh")
            questions = record.get("questions")
            if not isinstance(questions, list) or not questions:
                raise ValueError(f"{location}.questions must be a non-empty list")
            current_questions = []
            for question_index, question in enumerate(questions):
                current = self._question(question, f"{location}.questions[{question_index}]")
                if current["id"] in seen_question_ids:
                    raise ValueError(f"Duplicate question id: {current['id']!r}")
                seen_question_ids.add(current["id"])
                current_questions.append(current)
            revision = record.get("revision_metadata", {})
            if not isinstance(revision, dict):
                raise TypeError(f"{location}.revision_metadata must be a JSON object")
            regeneration = revision.get("require_regeneration", False)
            if not isinstance(regeneration, bool):
                raise ValueError(f"{location}.require_regeneration must be boolean")
            sample.update(
                questions=current_questions,
                require_regeneration=regeneration,
                question_bank_fingerprint=self._digest(current_questions),
                metadata_sha256=self.metadata_sha256,
            )
            records.append(sample)
        self.total_source_samples = len(records)
        self.total_source_questions = len(seen_question_ids)
        if self.start_index > len(records):
            raise IndexError(f"start_index={self.start_index} exceeds {len(records)} retained V3 records")
        records = records[self.start_index:]
        if self.max_samples is not None:
            records = records[:int(self.max_samples)]
        if not records:
            raise ValueError("No Light-Omni V3 samples remain after slicing")
        return records

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # A judge must not mutate the nested question bank through a sample view.
        return copy.deepcopy(self.samples[index])
