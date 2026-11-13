"""Image-only backends for the existing native draw tool."""
from __future__ import annotations

import base64
import hashlib
import importlib
import io
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image


class OpenAIImageProvider:
    def __init__(self, endpoint, model, api_key_env="OPTICS_IMAGE_API_KEY", allow_no_key=False,
                 size="1024x1024", seed=42, num_steps=50, **kwargs):
        import requests
        parts = urlsplit(endpoint)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("Image endpoint must be HTTP(S) without inline credentials/query")
        self.endpoint, self.model, self.size, self.seed = endpoint, model, size, int(seed)
        self.num_steps = int(num_steps)
        self.key = os.environ.get(api_key_env, "")
        if not self.key and not allow_no_key:
            raise ValueError("Missing image API key environment variable")
        self.session = requests.Session()
        self.last_metadata = None
        self.is_demo = False

    @property
    def identity(self):
        return {"provider": "openai_images", "endpoint": self.endpoint, "model": self.model,
                "size": self.size, "seed": self.seed, "num_steps": self.num_steps, "think": False}

    def generate(self, prompt, destination, timeout):
        self.last_metadata = None
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        payload = {"model": self.model, "prompt": prompt, "n": 1, "size": self.size,
                   "response_format": "b64_json", "seed": self.seed, "num_steps": self.num_steps,
                   "think": False}
        try:
            response = self.session.post(self.endpoint, json=payload, headers=headers, timeout=timeout, stream=True)
        except Exception as error:
            raise RuntimeError("Image request failed: " + type(error).__name__) from None
        try:
            if response.status_code != 200:
                raise RuntimeError(f"Invalid image response HTTP {response.status_code}")
            chunks, total = [], 0
            for chunk in response.iter_content(chunk_size=65536):
                total += len(chunk)
                if total > 80 * 1024 * 1024:
                    raise ValueError("Image response exceeds byte budget")
                chunks.append(chunk)
            import json
            body = json.loads(b"".join(chunks))
        finally:
            response.close()
        if len(body.get("data", [])) != 1 or not isinstance(body["data"][0].get("b64_json"), str):
            raise ValueError("Image server must return one base64 image; remote URL fetches are not implicit")
        raw = base64.b64decode(body["data"][0]["b64_json"], validate=True)
        if len(raw) > 50 * 1024 * 1024:
            raise ValueError("Image exceeds byte budget")
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            image = image.convert("RGB")
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, "PNG")
        self.last_metadata = {**self.identity, "usage": body.get("usage", {}),
                              "request_id": body.get("id"), "returned_image_sha256": hashlib.sha256(raw).hexdigest()}
        return self.last_metadata


class LocalSenseNovaProvider:
    def __init__(self, model_path, eval_repo, device="cuda:0", dtype="bf16", size="1024x1024",
                 seed=42, num_steps=50, cfg_scale=4.0, vram_mode="full", attn_backend="sdpa", **kwargs):
        self.model_path = str(Path(model_path).resolve())
        self.eval_repo = str(Path(eval_repo).resolve())
        self.device, self.dtype, self.size = device, dtype, size
        self.seed, self.num_steps, self.cfg_scale = int(seed), int(num_steps), float(cfg_scale)
        self.vram_mode, self.attn_backend = vram_mode, attn_backend
        self.client = None
        self.last_metadata = None
        self.is_demo = False

    @property
    def identity(self):
        return {"provider": "sensenova_local", "model": self.model_path, "size": self.size,
                "seed": self.seed, "num_steps": self.num_steps, "cfg_scale": self.cfg_scale, "think": False}

    def _load(self):
        if self.client is None:
            if not Path(self.eval_repo, "engine/client/sensenova_u1_client.py").is_file():
                raise ValueError("eval_repo must contain the audited SenseNova client")
            sys.path.insert(0, self.eval_repo)
            module = importlib.import_module("engine.client.sensenova_u1_client")
            actual = Path(module.__file__).resolve()
            if not actual.is_relative_to(Path(self.eval_repo)):
                raise ValueError("A different engine package is already imported")
            self.client = module.SenseNovaInferencer(model_path=self.model_path, device=self.device,
                dtype=self.dtype, vram_mode=self.vram_mode, attn_backend=self.attn_backend)

    def generate(self, prompt, destination, timeout):
        self._load()
        width, height = map(int, self.size.lower().replace("*", "x").split("x"))
        images = self.client.generate(prompt, image_size=(width, height), seed=self.seed,
            num_steps=self.num_steps, cfg_scale=self.cfg_scale, think=False)
        if len(images) != 1:
            raise ValueError("Expected one SenseNova image")
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        images[0].save(destination, "PNG")
        self.last_metadata = {**self.identity, "usage": {}}
        return self.last_metadata


def make_image_provider(cfg):
    values = dict(cfg)
    kind = values.pop("provider", "openai_images")
    if kind == "openai_images":
        return OpenAIImageProvider(**values)
    if kind == "sensenova_local":
        return LocalSenseNovaProvider(**values)
    if kind == "gpt_images":
        from .gpt_image_provider import GPTImageProvider
        return GPTImageProvider(**values)
    if kind == "wavespeed":
        from .wavespeed_image_provider import WaveSpeedImageProvider
        return WaveSpeedImageProvider(**values)
    if kind == "draw_align_fm":
        from .fm_image_provider import LocalFMCheckpointProvider
        return LocalFMCheckpointProvider(**values)
    if kind == "draw_align_checkpoint":
        from .checkpoint_image_provider import LocalCheckpointProvider
        return LocalCheckpointProvider(**values)
    raise ValueError("Image provider must be openai_images, sensenova_local, gpt_images, wavespeed, draw_align_fm or draw_align_checkpoint")
