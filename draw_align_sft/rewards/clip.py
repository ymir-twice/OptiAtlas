"""Frozen CLIP cosine with differentiable preprocessing and explicit long text.

Unlike upstream DiffusionOPSD's CLIP/PickScore truncation, chunk_mean scores
every text token in finite-context windows. This is a different, recorded reward
objective; it does not claim that CLIP can verify physical optical relationships.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from .features import projected_features


def _projected_features(output):
    return projected_features(output).float()


class CLIPScoreReward:
    def __init__(
        self, model_name="openai/clip-vit-large-patch14", *, revision=None,
        device="cpu", model=None, tokenizer=None, text_strategy="chunk_mean",
        window_overlap=0, image_strategy="full_resize", text_batch_size=32,
        cache_size=512, local_files_only=False,
    ):
        if (model is None) != (tokenizer is None):
            raise ValueError("inject both model and tokenizer, or neither")
        if model is None:
            from transformers import AutoTokenizer, CLIPModel
            model = CLIPModel.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
        self.model = model.to(device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.text_strategy = text_strategy
        self.image_strategy = image_strategy
        if text_strategy not in {"chunk_mean", "error"}:
            raise ValueError("text_strategy must be chunk_mean or error (silent truncation is unsupported)")
        if image_strategy not in {"full_resize", "center_crop"}:
            raise ValueError("image_strategy must be full_resize or center_crop")
        self.context_length = int(self.model.config.text_config.max_position_embeddings)
        self.image_size = int(self.model.config.vision_config.image_size)
        self.payload_length = self.context_length - tokenizer.num_special_tokens_to_add(pair=False)
        if not 0 <= window_overlap < self.payload_length:
            raise ValueError("window_overlap must be nonnegative and smaller than the text payload length")
        if text_batch_size < 1 or cache_size < 0:
            raise ValueError("text_batch_size must be positive and cache_size nonnegative")
        self.window_overlap = window_overlap
        self.text_batch_size = text_batch_size
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.last_metrics = {}
        self.metadata = {
            "kind": "clip_cosine", "model_name": model_name, "revision": revision,
            "text_strategy": text_strategy, "window_overlap": window_overlap,
            "context_length": self.context_length, "image_strategy": image_strategy,
            "image_size": self.image_size,
            "reward_scale": "unit-normalized image/text cosine, no learned logit scale",
            "reference_images": "unsupported; custom reward hook may implement reference conditioning",
        }

    def _token_windows(self, prompt):
        encoded = self.tokenizer(prompt, add_special_tokens=False, truncation=False)
        ids = encoded["input_ids"]
        if torch.is_tensor(ids):
            ids = ids.tolist()
        if len(ids) > self.payload_length and self.text_strategy == "error":
            raise ValueError(f"caption has {len(ids)} tokens but reward supports {self.payload_length}; choose chunk_mean")
        windows = []
        for start in range(0, max(1, len(ids)), self.payload_length - self.window_overlap):
            payload = ids[start:start + self.payload_length]
            build_specials = getattr(self.tokenizer, "build_inputs_with_special_tokens", None)
            if callable(build_specials):
                window = build_specials(payload)
            else:
                # Transformers 5 removed build_inputs_with_special_tokens.
                # CLIP's template remains BOS + payload + EOS, with no pair.
                bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
                if bos is None or eos is None or self.context_length-self.payload_length != 2:
                    raise ValueError("CLIP tokenizer must expose a two-token BOS/EOS template")
                window = [bos, *payload, eos]
            if len(window) > self.context_length:
                raise ValueError("tokenizer special-token accounting exceeds CLIP context length")
            windows.append(window)
            if start + self.payload_length >= len(ids):
                break
        return windows, len(ids)

    def _text_embeddings(self, prompt):
        if prompt in self._cache:
            self._cache.move_to_end(prompt)
            return self._cache[prompt]
        windows, token_count = self._token_windows(prompt)
        vectors = []
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        with torch.no_grad():
            for offset in range(0, len(windows), self.text_batch_size):
                batch = windows[offset:offset + self.text_batch_size]
                ids = torch.full((len(batch), self.context_length), pad_id, dtype=torch.long, device=self.device)
                mask = torch.zeros_like(ids)
                for index, window in enumerate(batch):
                    ids[index, :len(window)] = torch.tensor(window, dtype=torch.long, device=self.device)
                    mask[index, :len(window)] = 1
                features = self.model.get_text_features(input_ids=ids, attention_mask=mask)
                vectors.append(F.normalize(_projected_features(features), dim=-1))
        # Mean of per-window cosines: do NOT renormalize the mean vector. That
        # would change the objective and reward internally conflicting windows.
        result = (torch.cat(vectors).mean(0).detach(), len(windows), token_count)
        if self.cache_size:
            self._cache[prompt] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def _pixels(self, images):
        pixels = images.to(self.device).float()
        if self.image_strategy == "full_resize":
            pixels = F.interpolate(pixels, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False, antialias=True)
        else:
            height, width = pixels.shape[-2:]
            scale = self.image_size / min(height, width)
            height, width = max(self.image_size, round(height * scale)), max(self.image_size, round(width * scale))
            pixels = F.interpolate(pixels, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
            top, left = (height - self.image_size) // 2, (width - self.image_size) // 2
            pixels = pixels[..., top:top + self.image_size, left:left + self.image_size]
        mean = pixels.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = pixels.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        return ((pixels - mean) / std).to(dtype=next(self.model.parameters()).dtype)

    def __call__(self, images, prompts, reference_images=None):
        if reference_images is not None:
            raise ValueError("CLIP cosine does not use reference_images; supply a custom reference reward")
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] != len(prompts) or not len(prompts):
            raise ValueError("images must be [batch,3,height,width] and aligned with prompts")
        if not images.is_floating_point() or not torch.isfinite(images).all() or ((images < 0) | (images > 1)).any():
            raise ValueError("reward images must be finite floating tensors in [0,1]")
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError("reward prompts must be nonempty strings")
        text = [self._text_embeddings(prompt) for prompt in prompts]
        self.last_metrics = {
            "reward_text_windows_mean": sum(item[1] for item in text) / len(text),
            "reward_text_windows_max": max(item[1] for item in text),
            "reward_text_tokens_max": max(item[2] for item in text),
            "reward_text_truncated": 0,
        }
        # NO no_grad: model parameters are frozen, but the image/latent gradient
        # is required by DiffusionOPSD's target-construction phase.
        self.model.eval()
        image_features = self.model.get_image_features(pixel_values=self._pixels(images))
        image_vectors = F.normalize(_projected_features(image_features), dim=-1)
        text_vectors = torch.stack([item[0] for item in text])
        return (image_vectors * text_vectors).sum(-1).to(images.device)
