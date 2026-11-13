"""Lossless dense-caption manifests and a disjoint training/validation split."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
import os
from pathlib import Path
import uuid

import torch
from torch.utils.data import Dataset, Sampler

from ..data import ManifestPathResolver


def read_rows(path, split=None):
    path = Path(path).expanduser().resolve()
    rows, seen = [], set()
    resolve_image = ManifestPathResolver(path.parent)
    for number, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if split is not None and row.get("split", "train") != split:
            continue
        caption = row.get("caption", row.get("prompt"))
        identity = str(row.get("id", row.get("sha256", "")))
        if not identity or identity in seen or not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"{path}:{number}: unique id and nonempty complete caption required")
        seen.add(identity)
        image = row.get("image", row.get("target_image"))
        rows.append({**row, "id": identity, "caption": caption,
                     "image": str(resolve_image(image)) if image else None})
    if not rows:
        raise ValueError(f"No caption records in {path} (split={split!r})")
    return rows


def _keys(row):
    keys = {"caption:" + hashlib.sha256(row["caption"].encode()).hexdigest(), "id:" + row["id"]}
    # Paired-reward manifests name the same image-content identity differently
    # from legacy manifests. Copies at different paths must stay in one split,
    # including when train and validation use different hash field names.
    for field in ("sha256", "reference_image_sha256"):
        if row.get(field):
            keys.add("image_sha256:" + str(row[field]).lower())
    for field in ("image", "duplicate_group", "dedup_group", "group_id"):
        if row.get(field):
            keys.add(field + ":" + str(row[field]))
    return keys


def split_rows(config):
    rows = read_rows(config["data"]["path"], config["data"].get("split", "train"))
    options = config["upstream"]["validation"]
    enabled = options.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("upstream.validation.enabled must be a boolean")
    if not enabled:
        if int(config["upstream"].get("eval_freq", 0)) != 0:
            raise ValueError("Disabled validation requires upstream.eval_freq=0")
        if options.get("path"):
            raise ValueError("Disabled validation requires upstream.validation.path=null")
        return rows, []
    if options.get("path"):
        validation = read_rows(options["path"], options.get("split"))
        training_keys = set().union(*(_keys(r) for r in rows))
        if any(training_keys & _keys(r) for r in validation):
            raise ValueError("Training and validation overlap by caption, id, image path/hash, or saved duplicate group")
        return rows, validation
    # Union all known duplicate identifiers before splitting: the same caption
    # or image must never appear in both sets even when record IDs differ.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(rows):
        for key in _keys(row):
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    if len(groups) < 2:
        raise ValueError("At least two independent prompt/image groups are needed for held-out validation")
    fraction, limit = float(options["fraction"]), int(options["max_items"])
    if not 0 < fraction < 1 or limit < 1:
        raise ValueError("validation.fraction must be in (0,1), max_items must be positive")
    seed = str(options["seed"])
    ordered = sorted(groups.values(), key=lambda group: hashlib.sha256(
        (seed + min(r["caption"] for r in group)).encode()).hexdigest())
    target = min(limit, max(1, round(len(rows) * fraction)))
    validation, training = [], []
    for group in ordered[:-1]:
        (validation if len(validation) < target else training).extend(group)
    training.extend(ordered[-1])
    return training, validation


def materialize(config, directory):
    training, validation = split_rows(config)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", training), ("test", validation)):
        content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        tmp = directory / f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        tmp.write_text(content, encoding="utf8")
        tmp.replace(directory / f"{name}.jsonl")
    return training, validation


class TextPromptDataset(Dataset):
    """Upstream Dataset contract, preserving JSONL newlines and metadata."""
    def __init__(self, dataset, split="train"):
        directory = Path(dataset)
        path = directory / f"{split}.jsonl"
        if path.is_file():
            self.rows = read_rows(path)
        else:
            path = directory / f"{split}.txt"
            self.rows = [{"id": str(i), "caption": line.strip()}
                         for i, line in enumerate(path.read_text().splitlines()) if line.strip()]
        if not self.rows:
            raise ValueError(f"Empty dataset: {path}")
        self.file_path = str(path)
        self.prompts = [r["caption"] for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {"prompt": row["caption"], "metadata": {k: v for k, v in row.items() if k != "caption"}}

    @staticmethod
    def collate_fn(examples):
        return [r["prompt"] for r in examples], [r["metadata"] for r in examples]


class EpochKRepeatSampler(Sampler):
    """Opt-in complete prompt sweeps with the original K-repeat rank grouping.

    ``set_epoch`` takes the *global rollout batch index*, matching the trainer's
    ``outer_epoch * num_batches_per_epoch + batch``. Each data epoch uses one
    fixed-seed permutation; its final batch pads whole prompt groups from that
    permutation's start. No partial trajectory group or unequal rank batch is
    produced. Explicit seeds make a saved outer position sufficient to resume.
    """

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed):
        if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Epoch prompt sampling requires an explicit integer seed")
        for name, value in (("batch_size", batch_size), ("k", k), ("num_replicas", num_replicas)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must identify a policy replica")
        self.dataset, self.batch_size, self.k = dataset, batch_size, k
        self.num_replicas, self.rank, self.seed = num_replicas, rank, seed
        self.total_samples = num_replicas * batch_size
        if self.total_samples % k:
            raise ValueError("Global rollout batch must be divisible by K")
        self.m = self.total_samples // k
        if len(dataset) < self.m:
            raise ValueError("Dataset must contain at least one global batch of distinct prompt groups")
        self.batches_per_data_epoch = (len(dataset) + self.m - 1) // self.m
        self.epoch = 0
        self._permutation_epoch, self._permutation = None, None
        self._source_ids = None

    def batch_info(self, batch_index):
        if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
            raise ValueError("Global rollout batch index must be a nonnegative integer")
        data_epoch, position = divmod(batch_index, self.batches_per_data_epoch)
        if data_epoch != self._permutation_epoch:
            generator = torch.Generator().manual_seed(self.seed + data_epoch)
            self._permutation = torch.randperm(len(self.dataset), generator=generator).tolist()
            self._permutation_epoch = data_epoch
        indices = self._permutation[position * self.m:(position + 1) * self.m]
        padding = self.m - len(indices)
        if padding:
            indices += self._permutation[:padding]
        return {"indices": indices, "padding_groups": padding, "data_epoch": data_epoch}

    def set_epoch(self, epoch):
        self.batch_info(epoch)  # Validate the saved/global batch position.
        self.epoch = epoch

    def __iter__(self):
        while True:
            indices = self.batch_info(self.epoch)["indices"]
            repeated = [index for index in indices for _ in range(self.k)]
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(repeated), generator=generator).tolist()
            shuffled = [repeated[index] for index in order]
            start = self.rank * self.batch_size
            yield shuffled[start:start + self.batch_size]


def write_epoch_coverage(directory, sampler, prompt_ids, *, outer_epoch, global_step, num_batches):
    """Append a rank-zero record after the outer optimizer/behavior update.

    Validate actual gathered source IDs against the sampler, including K and
    padding. Cumulative counts derive from the saved outer position, so a
    resumed run (or a replayed round after an interrupted save) cannot inflate
    coverage by appending duplicate log entries. Only this round's IDs are
    written, rather than rewriting an ever-growing list on every update.
    """
    if sampler._source_ids is None:
        sampler._source_ids = [str(sampler.dataset[index]["metadata"]["id"])
                               for index in range(len(sampler.dataset))]
        if len(set(sampler._source_ids)) != len(sampler.dataset):
            raise ValueError("Coverage requires unique dataset source IDs")
    start, end = outer_epoch * num_batches, (outer_epoch + 1) * num_batches
    expected, padding_ids = [], []
    for batch in range(start, end):
        info = sampler.batch_info(batch)
        ids = [sampler._source_ids[index] for index in info["indices"]]
        expected.extend(identity for identity in ids for _ in range(sampler.k))
        if info["padding_groups"]:
            padding_ids.extend(ids[-info["padding_groups"]:])
    if Counter(prompt_ids) != Counter(expected):
        raise ValueError("Actual sampled source IDs do not match the completed epoch-sampler batches")
    total = len(sampler.dataset)
    completed, partial_batches = divmod(end, sampler.batches_per_data_epoch)
    covered = min(total, end * sampler.m)
    record = {
        "format": 1, "sampling": "epoch", "status": "optimizer_completed",
        "epoch_completed": outer_epoch + 1, "global_step": global_step,
        "global_batch_start": start, "global_batch_end_exclusive": end,
        "source_ids": list(dict.fromkeys(prompt_ids)),
        "padding_source_ids": padding_ids, "padding_prompt_groups": len(padding_ids),
        "rollout_trajectories": len(prompt_ids), "sampled_prompt_groups": num_batches * sampler.m,
        "training_records": total, "unique_records_covered": covered,
        "coverage_fraction": covered / total, "data_epochs_completed": completed,
        "records_in_current_data_epoch": min(total, partial_batches * sampler.m),
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "coverage.jsonl").open("a", encoding="utf8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
