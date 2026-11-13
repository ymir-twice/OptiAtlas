"""FrozenDrawClient-compatible portable drawing HTTP boundary."""
from __future__ import annotations

import base64

from contextlib import ExitStack, nullcontext

import fcntl

import hmac

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import json

import os

from pathlib import Path

import signal

import socket

import subprocess

import sys

import tempfile

import threading

import time

import traceback

import urllib.request

from .network import host as validate_host, resolve_host, family, socket_for, http_url, health_host

def _event(name, **fields):
    print(json.dumps({"event": name, **fields}, allow_nan=False), flush=True)

def create_server(cfg, provider, *, auxiliary_gate=None):
    """Expose the native provider with the existing FrozenDrawClient protocol."""
    key = os.environ.get(cfg.get("api_key_env", "OPTICS_IMAGE_API_KEY"), "")
    if not key and not cfg.get("allow_no_key", False):
        raise ValueError("Drawing service API key is missing")
    slots = threading.BoundedSemaphore(int(cfg["max_pending_requests"]))
    lock = threading.Lock()
    model = cfg["model_name"]
    image = cfg["image"]
    completed = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, body):
            raw = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def authorized(self):
            if key and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key):
                self.respond(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path != "/health":
                return self.respond(404, {"error": "not_found"})
            self.respond(200, {"status": "ready", "model": model, "completed": completed,
                               "size": image["size"], "num_steps": image["num_steps"],
                               "cfg_scale": image["cfg_scale"], "identity": provider.identity})

        def do_POST(self):
            nonlocal completed
            if not self.authorized():
                return
            if self.path != "/v1/images/generations":
                return self.respond(404, {"error": "not_found"})
            try:
                count = int(self.headers.get("Content-Length", "0"))
                if not 0 < count <= 256 * 1024:
                    raise ValueError("Invalid request size")
                self.connection.settimeout(float(cfg["read_timeout_seconds"]))
                request = json.loads(self.rfile.read(count))
                prompt = request["prompt"]
                seed = request.get("seed", image["seed"])
                if not isinstance(prompt, str) or not 0 < len(prompt.strip()) <= 20000:
                    raise ValueError("Invalid prompt")
                if type(seed) is not int or not 0 <= seed < 2**63:
                    raise ValueError("Invalid seed")
                if (request.get("model") != model or request.get("n", 1) != 1
                        or request.get("think", False) is not False
                        or request.get("response_format", "b64_json") != "b64_json"
                        or request.get("size", image["size"]) != image["size"]
                        or request.get("num_steps", image["num_steps"]) != image["num_steps"]
                        or request.get("cfg_scale", image["cfg_scale"]) != image["cfg_scale"]):
                    raise ValueError("Request differs from frozen generation configuration")
            except (ValueError, KeyError, TypeError, AttributeError, OSError):
                return self.respond(400, {"error": "invalid_request"})
            if not slots.acquire(blocking=False):
                return self.respond(429, {"error": "queue_full"})
            started = time.monotonic()
            try:
                # Local temporary PNGs are removed after returning their bytes.
                # Mark pending before the existing thread lock: queued draw
                # requests must also outrank the auxiliary's next mini-step.
                with (auxiliary_gate.pending_draw() if auxiliary_gate else nullcontext()), lock, \
                        (auxiliary_gate.gpu() if auxiliary_gate else nullcontext()), \
                        tempfile.TemporaryDirectory(prefix="optics-rl-draw-") as directory:
                    target = Path(directory) / "image.png"
                    provider._apply_seed(seed)
                    provider.generate(prompt, target, timeout=None)
                    encoded = base64.b64encode(target.read_bytes()).decode("ascii")
                    completed += 1
                    request_id = f"draw-{os.getpid()}-{completed}"
                    _event("draw_completed", completed=completed, seconds=round(time.monotonic() - started, 2), seed=seed)
                self.respond(200, {"id": request_id, "model": model,
                                   "data": [{"b64_json": encoded}], "usage": {"images": 1}})
            except (BrokenPipeError, ConnectionResetError):
                _event("draw_client_disconnected")
            except Exception as error:
                traceback.print_exc()
                self.respond(500, {"error": "generation_failed", "error_type": type(error).__name__})
            finally:
                slots.release()

    class DrawingHTTPServer(ThreadingHTTPServer):
        address_family = family(cfg["host"])

    server = DrawingHTTPServer((cfg["host"], int(cfg["port"])), Handler)
    server.daemon_threads = True
    return server

