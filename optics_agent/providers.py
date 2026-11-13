"""Text-to-image provider and safe transport; no automatic credential discovery.

The draw tool uses Qwen-Image-2.0 native DashScope synchronous text-to-image.
Native multimodal teacher/student chat lives in llm.py.
"""
import base64
import errno
import hashlib
import io
import ipaddress
import json
import math
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .tls import create_ssl_context


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024


class ImageProvider(Protocol):
    """Minimal image interface consumed by draw and the prompt-only baseline.

    Implementations write a PNG to destination and return safe usage/request
    metadata. The explicit is_demo marker is checked separately by the runner.
    """
    @property
    def identity(self) -> dict[str, Any]: ...

    def generate(self, prompt: str, destination: Path, timeout: float) -> dict[str, Any]: ...


class ProviderError(RuntimeError):
    """Safe, structured failure; never carries raw headers, bodies or signed URLs."""
    def __init__(self, code, request_id=None, *, error_type=None):
        self.code = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(code))[:100]
        self.request_id = _identifier(request_id)
        self.error_type = (re.sub(r"[^A-Za-z0-9_]", "_", str(error_type))[:100]
                           if error_type else None)
        super().__init__(self.code)


def _identifier(value):
    return re.sub(r"[^A-Za-z0-9_.:/-]", "_", str(value))[:200] if value else None


def _endpoint(url, allow_loopback=True):
    parsed = urllib.parse.urlsplit(url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
        raise ValueError("Endpoint must have a hostname and no credentials, query or fragment")
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (allow_loopback and local and parsed.scheme == "http"):
        raise ValueError("Endpoint requires HTTPS (or HTTP on localhost for a future local model)")
    return url


def _download_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ProviderError("invalid_image_download_url")
    if parsed.hostname in {"localhost", "localhost.localdomain"}:
        raise ProviderError("invalid_image_download_host")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ProviderError("invalid_image_download_host")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError("redirect_not_allowed")


def _transport_failure(error, fallback="transport_error"):
    """Classify typed failures without retaining exception messages or URLs."""
    current = error
    seen = set()
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        code = None
        if isinstance(current, ssl.SSLCertVerificationError):
            code = "tls_certificate_error"
        elif isinstance(current, ssl.SSLError):
            code = "tls_error"
        elif isinstance(current, socket.gaierror):
            code = "dns_error"
        elif getattr(current, "errno", None) in {errno.ENETUNREACH, errno.EHOSTUNREACH}:
            code = "network_unreachable"
        elif isinstance(current, ConnectionRefusedError) or getattr(current, "errno", None) == errno.ECONNREFUSED:
            code = "connection_refused"
        elif isinstance(current, TimeoutError) or getattr(current, "errno", None) == errno.ETIMEDOUT:
            code = "request_timeout"
        elif isinstance(current, ConnectionResetError) or getattr(current, "errno", None) == errno.ECONNRESET:
            code = "connection_reset"
        if code:
            return ProviderError(code, error_type=type(current).__name__)
        nested = (current.reason if isinstance(current, urllib.error.URLError)
                  and isinstance(current.reason, BaseException)
                  else current.__cause__ or current.__context__)
        if not isinstance(nested, BaseException) or id(nested) in seen:
            break
        current = nested
    return ProviderError(fallback, error_type=type(current).__name__)


def transport_error_code(error):
    """Public, safe diagnostic code for a failed network operation."""
    return _transport_failure(error).code


class HTTPTransport:
    """No retries: a timed out image POST may already have incurred a charge."""
    def __init__(self):
        self.opener = urllib.request.build_opener(
            _NoRedirect(), urllib.request.HTTPSHandler(context=create_ssl_context()))

    def post_json(self, url, headers, payload, timeout):
        request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                         headers=headers, method="POST")
        try:
            with self.opener.open(request, timeout=timeout) as response:
                data = response.read(MAX_JSON_BYTES + 1)
                if len(data) > MAX_JSON_BYTES:
                    raise ProviderError("response_too_large")
                response_headers = dict(response.headers)
            return json.loads(data), response_headers
        except ProviderError:
            raise
        except urllib.error.HTTPError as exc:
            raise ProviderError("http_" + str(exc.code), exc.headers.get("x-request-id")) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise _transport_failure(exc) from None
        except (ValueError, UnicodeError):
            raise ProviderError("invalid_json_response") from None

    def download(self, url, timeout):
        _download_url(url)
        try:
            with self.opener.open(urllib.request.Request(url, method="GET"), timeout=timeout) as response:
                data = response.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                raise ProviderError("image_too_large")
            return data
        except ProviderError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise _transport_failure(exc, fallback="image_download_error") from None


def _env_key(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
        raise ValueError("api_key_env must be an uppercase environment variable name")
    value = os.environ.get(name)
    if not value:
        raise ValueError("Missing explicitly configured environment variable: " + name)
    return value


def safe_media_path(run_dir, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Media paths must be relative to the run directory")
    root = Path(run_dir).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Media path is outside run directory or missing")
    return target


def image_data_uri(path, *, expected_sha256=None):
    """Encode exactly the bytes whose hash was archived, if a binding is supplied."""
    from PIL import Image
    path = Path(path)
    data = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ProviderError("input_image_hash_changed")
    if len(data) > MAX_IMAGE_BYTES:
        raise ProviderError("image_too_large")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
            mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(img.format)
        if mime is None:
            raise ProviderError("unsupported_image_format")
    except (OSError, ValueError):
        raise ProviderError("invalid_image_bytes") from None
    return "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")


def _metadata(response, headers, requested_model, provider):
    # Do not persist reasoning_content, API bodies, keys or signed result URLs.
    usage = response.get("usage", {})
    return {"provider": provider, "requested_model": requested_model,
            "served_model": _identifier(response.get("model")),
            "request_id": _identifier(response.get("request_id") or response.get("id")
                                      or headers.get("x-request-id") or headers.get("X-Request-Id")),
            "usage": _numeric_usage(usage)}


def _numeric_usage(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[str(key)] = _numeric_usage(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            result[str(key)] = item
    return result


class DashScopeImageProvider:
    def __init__(self, generation_url, model="qwen-image-2.0-2026-03-03",
                 api_key_env="OPTICS_IMAGE_API_KEY", transport=None,
                 size="1024*1024", seed=113):
        self.url = _endpoint(generation_url)
        self.model = str(model)
        self.key = _env_key(api_key_env)
        self.transport = transport or HTTPTransport()
        if not re.fullmatch(r"\d+\*\d+", size):
            raise ValueError("size must be WIDTH*HEIGHT")
        width, height = map(int, size.split("*"))
        if min(width, height) < 1 or not 512**2 <= width * height <= 2048**2:
            raise ValueError("Qwen-Image-2.0 pixel count must be between 512² and 2048²")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2147483647:
            raise ValueError("seed must be an integer from 0 to 2147483647")
        self.size, self.seed = size, seed
        self.last_metadata = None

    @property
    def identity(self):
        return {"provider": "dashscope_native", "model": self.model, "endpoint": self.url,
                "size": self.size, "seed": self.seed, "prompt_extend": False}

    def generate(self, prompt, destination, timeout):
        self.last_metadata = None
        return self._request([{"text": prompt}], destination, timeout)

    def _request(self, content, destination, timeout):
        from PIL import Image
        deadline = time.monotonic() + timeout
        payload = {"model": self.model, "input": {"messages": [{"role": "user", "content": content}]},
                   "parameters": {"n": 1, "size": self.size, "seed": self.seed,
                                  "prompt_extend": False, "watermark": False}}
        response, headers = self.transport.post_json(
            self.url, {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
            payload, timeout)
        if not isinstance(response, dict):
            raise ProviderError("invalid_image_response")
        meta = _metadata(response, headers, self.model, "dashscope_native")
        self.last_metadata = meta
        if response.get("code") or response.get("error"):
            raise ProviderError("image_provider_rejected_request", meta["request_id"])
        try:
            contents = response["output"]["choices"][0]["message"]["content"]
            urls = [item["image"] for item in contents if "image" in item]
            if len(urls) != 1:
                raise ValueError()
            _download_url(urls[0])
        except (KeyError, IndexError, TypeError, ValueError):
            raise ProviderError("invalid_image_response", meta["request_id"]) from None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderError("image_deadline_exceeded", meta["request_id"])
        try:
            data = self.transport.download(urls[0], remaining)
        except ProviderError as exc:
            raise ProviderError(exc.code, meta["request_id"]) from None
        if len(data) > MAX_IMAGE_BYTES:
            raise ProviderError("image_too_large", meta["request_id"])
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
                if img.format != "PNG":
                    raise ValueError()
        except (OSError, ValueError):
            raise ProviderError("invalid_generated_png", meta["request_id"]) from None
        Path(destination).write_bytes(data)
        return meta
