"""Light-Omni text-to-image prompt dataset."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import Dataset


class LightOmniDataset(Dataset):
    """Load JSONL/JSON prompts, preferring an explicit generation prompt."""

    REQUIRED_FIELDS = ("category_zh", "prompt_zh", "prompt_en")

    def __init__(
        self,
        metadata_file: str | Path = (
            "data/light-omni/optical_phenomena_prompts_bilingual.jsonl"
        ),
        start_index: int = 0,
        max_samples: Optional[int] = None,
    ):
        self.metadata_file = Path(metadata_file)
        self.start_index = int(start_index)
        self.max_samples = max_samples
        if not self.metadata_file.is_file():
            raise FileNotFoundError(
                f"Light-Omni metadata file not found: {self.metadata_file}"
            )
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if max_samples is not None and int(max_samples) < 0:
            raise ValueError("max_samples must be non-negative or null")
        self.samples = self._load()

    def _load(self) -> list[Dict[str, Any]]:
        raw_records = self._read_records()
        records: list[Dict[str, Any]] = []
        seen_sample_ids: set[str] = set()
        for record_index, record in enumerate(raw_records):
            location = f"record {record_index}"
            missing = [
                field for field in self.REQUIRED_FIELDS if field not in record
            ]
            if missing:
                raise KeyError(f"{location} is missing fields: {missing}")
            for field in self.REQUIRED_FIELDS:
                if (
                    not isinstance(record[field], str)
                    or not record[field].strip()
                ):
                    raise ValueError(
                        f"Field {field!r} in {location} must be a "
                        "non-empty string"
                    )
            prompt_en = record["prompt_en"].strip()
            generation_prompt = record.get("generation_prompt", prompt_en)
            if (
                not isinstance(generation_prompt, str)
                or not generation_prompt.strip()
            ):
                raise ValueError(
                    f"generation_prompt in {location} must be a non-empty string"
                )
            generation_prompt = generation_prompt.strip()
            # The generated image belongs to the effective prompt. Hashing only
            # prompt_en would reuse an old image ID after generation_prompt changes.
            digest = hashlib.sha1(generation_prompt.encode("utf-8")).hexdigest()[:12]
            raw_sample_id = record.get(
                "sample_id", f"{record_index:06d}_{digest}"
            )
            if not isinstance(raw_sample_id, str) or not raw_sample_id.strip():
                raise ValueError(f"sample_id in {location} must be a string")
            sample_id = self._filename_sample_id(raw_sample_id.strip())
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id: {sample_id!r}")
            seen_sample_ids.add(sample_id)
            category_id = record.get("category_id")
            category_en = record.get("category_en", category_id)
            if not isinstance(category_en, str) or not category_en.strip():
                raise ValueError(
                    f"{location} needs a non-empty category_en or category_id"
                )
            records.append(
                {
                    "source_index": record_index,
                    "sample_id": sample_id,
                    "category_id": (
                        category_id.strip()
                        if isinstance(category_id, str)
                        else None
                    ),
                    "category_en": category_en.strip(),
                    "category_zh": record["category_zh"].strip(),
                    "prompt_en": prompt_en,
                    "prompt_zh": record["prompt_zh"].strip(),
                    "generation_prompt": generation_prompt,
                }
            )
        if self.start_index > len(records):
            raise IndexError(
                f"start_index={self.start_index} exceeds {len(records)} records"
            )
        records = records[self.start_index :]
        if self.max_samples is not None:
            records = records[: int(self.max_samples)]
        if not records:
            raise ValueError("No Light-Omni samples remain after slicing")
        return records

    @staticmethod
    def _filename_sample_id(raw_sample_id: str) -> str:
        """Keep existing safe IDs, disambiguating lossy/overlong filenames."""
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_sample_id)
        # Leave room for extensions and atomic-save suffixes on filesystems
        # with a 255-byte filename limit. ASCII output also fixes byte length.
        if sanitized != raw_sample_id or len(sanitized) > 160 or sanitized in {".", ".."}:
            digest = hashlib.sha1(raw_sample_id.encode("utf-8")).hexdigest()[:12]
            sanitized = f"{sanitized[:147]}_{digest}"
        return sanitized

    def _read_records(self) -> list[Dict[str, Any]]:
        """Accept a JSON array as well as genuine one-object-per-line JSONL."""
        # Spreadsheet/editor exports can include a UTF-8 BOM before a JSON
        # array or the first JSONL record.
        text = self.metadata_file.read_text(encoding="utf-8-sig")
        if not text.strip():
            raise ValueError(f"Empty Light-Omni metadata: {self.metadata_file}")
        if text.lstrip().startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON array in {self.metadata_file}: {error}"
                ) from error
            if not isinstance(value, list):
                raise TypeError("Light-Omni JSON root must be an array")
            raw_records = value
        else:
            raw_records = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {self.metadata_file}:{line_number}: "
                        f"{error}"
                    ) from error
        for index, record in enumerate(raw_records):
            if not isinstance(record, dict):
                raise TypeError(f"Light-Omni record {index} must be a JSON object")
        return raw_records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return dict(self.samples[index])
