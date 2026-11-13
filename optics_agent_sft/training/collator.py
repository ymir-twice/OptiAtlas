"""Exact next-decision loss masking; no truncation and no template guessing."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from .template import (QWEN35_MODEL_TYPES, ordered_image_paths, processor_messages,
                       reject_reserved_markup)


def completion_labels(input_ids: list[int], attention_mask: list[int], prefix_ids: list[int],
                      image_token_ids: set[int]) -> list[int]:
    """Mask every context/pad/visual token, including previous assistant actions.

    Padding is identified by attention_mask (not equality with pad_token_id),
    so an EOS token remains supervised even when the tokenizer uses EOS as pad.
    """
    if len(input_ids) != len(attention_mask):
        raise ValueError("Token/mask length mismatch")
    positions = [i for i, active in enumerate(attention_mask) if active]
    sequence = [input_ids[i] for i in positions]
    if sequence[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Processor prefix mismatch; refusing approximate assistant loss mask")
    if len(sequence) <= len(prefix_ids):
        raise ValueError("No completion tokens remain")
    labels = [-100] * len(input_ids)
    for i in positions[len(prefix_ids):]:
        if input_ids[i] not in image_token_ids:
            labels[i] = input_ids[i]
    if all(label == -100 for label in labels):
        raise ValueError("Completion has no trainable text tokens")
    return labels


class DecisionCollator:
    def __init__(self, processor, *, max_length: int = 16384, image_token_ids=None,
                 image_loader=None, token_validation_cache_size: int = 128,
                 reuse_image_preprocessing: bool = True, model_type: str | None = None):
        if (isinstance(token_validation_cache_size, bool)
                or not isinstance(token_validation_cache_size, int)
                or token_validation_cache_size < 0):
            raise ValueError("token_validation_cache_size must be a nonnegative integer")
        self.processor = processor
        self.model_type = model_type
        # Our reviewed trajectories contain public assistant text/actions, not
        # private chain-of-thought. Match Qwen3.5's non-thinking generation prefix
        # to its full-message template; the automatic empty <think> block is
        # prefix context and must never become a fabricated training target.
        self.chat_template_kwargs = {"enable_thinking": False} if model_type in QWEN35_MODEL_TYPES else {}
        self.max_length = max_length
        self.image_loader = image_loader or self._load_image
        self.token_validation_cache_size = token_validation_cache_size
        if not isinstance(reuse_image_preprocessing, bool):
            raise ValueError("reuse_image_preprocessing must be boolean")
        self.reuse_image_preprocessing = reuse_image_preprocessing
        # Only immutable CPU token IDs are retained: never images, pixel tensors,
        # model outputs or CUDA allocations. Every batch still processes its pixels.
        self._token_cache = OrderedDict()
        self._cache_hits = self._cache_misses = 0
        self._processor_objects = (id(processor), id(processor.tokenizer))
        self._processor_config = self._processor_signature()
        self.image_token_ids = set(image_token_ids or ())
        tokenizer = processor.tokenizer
        for token in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                self.image_token_ids.add(token_id)

    @staticmethod
    def _digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                         default=str).encode("utf-8")).hexdigest()

    def _processor_signature(self):
        """Configuration identity within this collator, without serializing weights/vocab."""
        processor, tokenizer = self.processor, self.processor.tokenizer
        config = {"processor_class": type(processor).__qualname__,
                  "model_type": self.model_type,
                  "chat_template_kwargs": self.chat_template_kwargs,
                  "image_token": getattr(processor, "image_token", None),
                  "chat_template": getattr(processor, "chat_template", None),
                  "tokenizer_class": type(tokenizer).__qualname__}
        for name in ("init_kwargs", "chat_template", "padding_side", "truncation_side",
                     "special_tokens_map", "vocab_size", "model_max_length"):
            config["tokenizer_" + name] = getattr(tokenizer, name, None)
        if hasattr(tokenizer, "get_added_vocab"):
            config["added_vocab"] = tokenizer.get_added_vocab()
        for name in ("image_processor", "video_processor"):
            component = getattr(processor, name, None)
            if component is not None:
                config[name] = {"class": type(component).__qualname__,
                                "config": component.to_dict() if hasattr(component, "to_dict") else str(component)}
        return self._digest(config)

    def cache_info(self):
        return {"hits": self._cache_hits, "misses": self._cache_misses,
                "size": len(self._token_cache), "capacity": self.token_validation_cache_size}

    def __getstate__(self):
        # Spawned DataLoader workers own their processor/cache. Avoid serializing
        # a warm cache into every worker, and never compare parent process IDs.
        state = self.__dict__.copy()
        state.update(_token_cache=OrderedDict(), _cache_hits=0, _cache_misses=0,
                     _processor_objects=None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._processor_objects = (id(self.processor), id(self.processor.tokenizer))
        self._processor_config = self._processor_signature()

    @staticmethod
    def _image_digest(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _file_state(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @staticmethod
    def _load_image(path):
        from PIL import Image
        with Image.open(path) as source:
            return source.convert("RGB")

    def _process(self, texts, images):
        kwargs = {"text": texts, "padding": True, "truncation": False,
                  "return_tensors": "pt", "add_special_tokens": False}
        if images:
            kwargs["images"] = images
        return self.processor(**kwargs)

    def _can_reuse_image_preprocessing(self):
        # This expansion is checked against the actual batched processor below.
        # Unknown/custom processors keep the general, independently encoded path.
        return (self.reuse_image_preprocessing
                and type(self.processor).__module__ == "transformers.models.qwen3_vl.processing_qwen3_vl"
                and type(self.processor).__name__ == "Qwen3VLProcessor")

    def _expand_qwen_images(self, text, grids):
        """Use the real batch's grids, never infer visual token counts from pixels.

        Qwen3-VL expands each image placeholder to prod(grid) / merge_size**2
        tokens. Re-encoding these expanded strings without images preserves the
        full/prefix token check without preprocessing the same pixels three times.
        The original batch remains authoritative for every multimodal tensor.
        """
        import math

        token = self.processor.image_token
        chunks = text.split(token)
        if len(chunks) != len(grids) + 1:
            raise ValueError("Qwen image grid count differs from rendered placeholders")
        merge = self.processor.image_processor.merge_size
        if type(merge) is not int or merge <= 0:
            raise ValueError("Qwen image merge_size must be a positive integer")
        result = [chunks[0]]
        for grid, chunk in zip(grids, chunks[1:]):
            if len(grid) != 3 or any(type(size) is not int or size <= 0 for size in grid):
                raise ValueError("Qwen image grids require three positive integer dimensions")
            patches = math.prod(grid)
            if patches % (merge * merge):
                raise ValueError("Qwen image grid is not divisible by spatial merge size")
            result.extend((token * (patches // (merge * merge)), chunk))
        return "".join(result)

    def _validated_ids(self, sample, prefix_text, full_text, images, cache_key, *, grids=None):
        cached = self._token_cache.get(cache_key)
        if cached is None:
            self._cache_misses += 1
            if grids is None:
                prefix = self._process([prefix_text], images)
                single = self._process([full_text], images)
                prefix_ids = prefix["input_ids"][0][prefix["attention_mask"][0].bool()].tolist()
                full_ids = single["input_ids"][0][single["attention_mask"][0].bool()].tolist()
            else:
                text_only = self._process([self._expand_qwen_images(text, grids)
                                           for text in (prefix_text, full_text)], [])
                prefix_ids, full_ids = [ids[mask.bool()].tolist() for ids, mask in
                                        zip(text_only["input_ids"], text_only["attention_mask"])]
        else:
            self._cache_hits += 1
            self._token_cache.move_to_end(cache_key)
            prefix_ids, full_ids = map(list, cached)
        completion_labels(full_ids, [1] * len(full_ids), prefix_ids, self.image_token_ids)
        if len(full_ids) > self.max_length:
            raise ValueError(f"{sample.get('sample_id')}: {len(full_ids)} tokens exceed max_length "
                             f"{self.max_length}; shorten context explicitly, never truncate image tokens")
        if cached is None and self.token_validation_cache_size:
            self._token_cache[cache_key] = (tuple(prefix_ids), tuple(full_ids))
            while len(self._token_cache) > self.token_validation_cache_size:
                self._token_cache.popitem(last=False)
        return prefix_ids, full_ids

    def __call__(self, samples):
        import torch
        if (id(self.processor), id(self.processor.tokenizer)) != self._processor_objects:
            raise ValueError("Processor/tokenizer replacement requires a new collator")
        signature = self._processor_signature()
        if signature != self._processor_config:
            self._token_cache.clear()
            self._processor_config = signature
        # Rehash actual files once per invocation, rather than trusting a stale
        # stat or the curated digest alone across epochs. This dictionary is local.
        image_digests = {}
        full_texts, flat_images, prefixes, singles = [], [], [], []
        pending = []
        reuse_images = self._can_reuse_image_preprocessing()
        for sample in samples:
            if sample.get("schema_version") != 2:
                raise ValueError("Only native schema 2 conversations may be collated")
            target = sample.get("target_message_index")
            if not isinstance(target, int) or target != len(sample["messages"]) - 1:
                raise ValueError("The selected assistant target must be the last conversation message")
            messages = processor_messages(sample["messages"][:target], model_type=self.model_type)
            completion = processor_messages(sample["messages"][target:], model_type=self.model_type)[0]
            if completion.get("role") != "assistant":
                raise ValueError("Supervised target must be an assistant message")
            tools = sample["tools"]
            reject_reserved_markup(tools)
            image_paths = ordered_image_paths(messages)
            expected_hashes = sample.get("image_sha256")
            if expected_hashes is not None and (not isinstance(expected_hashes, list)
                                                or len(expected_hashes) != len(image_paths)):
                raise ValueError("Curated image SHA256 order/count does not match context images")
            image_keys = []
            images = []
            for index, path in enumerate(image_paths):
                path = path.resolve()
                if path not in image_digests:
                    state = self._file_state(path)
                    digest = self._image_digest(path)
                    if self._file_state(path) != state:
                        raise ValueError("Context image changed while reading its SHA256")
                    image_digests[path] = digest, state
                digest, state = image_digests[path]
                if expected_hashes is not None and expected_hashes[index] != digest:
                    raise ValueError("Context image SHA256 changed after admission")
                if self._file_state(path) != state:
                    raise ValueError("Context image changed during batch preparation")
                images.append(self.image_loader(path))
                if self._file_state(path) != state:
                    raise ValueError("Context image changed during batch preparation")
                image_keys.append((str(path), digest))
            prefix_text = self.processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                                             **self.chat_template_kwargs)
            without_tools = self.processor.apply_chat_template(messages, tools=None, tokenize=False, add_generation_prompt=True,
                                                               **self.chat_template_kwargs)
            if tools and prefix_text == without_tools:
                raise ValueError("Chat template ignored native tool definitions")
            full_messages = messages + [completion]
            full_text = self.processor.apply_chat_template(full_messages, tools=tools, tokenize=False, add_generation_prompt=False,
                                                           **self.chat_template_kwargs)
            if completion.get("tool_calls"):
                without_call = copy.deepcopy(full_messages)
                without_call[-1].pop("tool_calls", None)
                call_free_text = self.processor.apply_chat_template(without_call, tools=tools, tokenize=False, add_generation_prompt=False,
                                                                    **self.chat_template_kwargs)
                if full_text == call_free_text:
                    raise ValueError("Chat template ignored native assistant tool_calls")
            image_token = getattr(self.processor, "image_token", "<|image_pad|>")
            if prefix_text.count(image_token) != len(images) or full_text.count(image_token) != len(images):
                raise ValueError("Chat template dropped or duplicated image placeholders; refusing hidden-image SFT")
            cache_key = self._digest({"processor": signature, "prefix": prefix_text,
                                      "full": full_text, "tools": tools, "images": image_keys})
            full_texts.append(full_text)
            flat_images.extend(images)
            if reuse_images:
                pending.append((sample, prefix_text, full_text, images, cache_key))
            else:
                prefix_ids, full_ids = self._validated_ids(sample, prefix_text, full_text, images, cache_key)
                prefixes.append(prefix_ids)
                singles.append(full_ids)
        batch = self._process(full_texts, flat_images)
        if reuse_images:
            grid_tensor = batch.get("image_grid_thw")
            grids = grid_tensor.tolist() if grid_tensor is not None else []
            if len(grids) != len(flat_images):
                raise ValueError("Qwen batch image grid count differs from context images")
            offset = 0
            for sample, prefix_text, full_text, images, cache_key in pending:
                image_grids = grids[offset:offset + len(images)]
                prefix_ids, full_ids = self._validated_ids(
                    sample, prefix_text, full_text, images, cache_key, grids=image_grids)
                prefixes.append(prefix_ids)
                singles.append(full_ids)
                offset += len(images)
        labels = []
        for index, prefix_ids in enumerate(prefixes):
            ids = batch["input_ids"][index].tolist()
            attention = batch["attention_mask"][index].tolist()
            if [token for token, active in zip(ids, attention) if active] != singles[index]:
                raise ValueError("Batched processor changed an example's tokens/image expansion")
            labels.append(completion_labels(ids, attention, prefix_ids, self.image_token_ids))
        # Qwen documentation explicitly removes token_type_ids; multimodal tensors
        # such as pixel_values and image_grid_thw must remain untouched.
        batch.pop("token_type_ids", None)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
