"""Public request conventions shared by company and local chat API adapters.

No credentials are read here. The company deployment URL and token conventions
match ``call_company_model.py``; secrets remain in the calling process environment.
"""
from __future__ import annotations

import math
import os
from urllib.parse import parse_qsl, quote, urlsplit


def validate_chat_endpoint(endpoint: str) -> str:
    """Allow the public Azure-style api-version query, never credential queries."""
    if not isinstance(endpoint, str):
        raise ValueError("Chat endpoint must be an HTTP(S) URL")
    parts = urlsplit(endpoint)
    query = parse_qsl(parts.query, keep_blank_values=True)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username
            or parts.password or parts.fragment or len(query) > 1
            or any(key != "api-version" or not value for key, value in query)):
        raise ValueError("Chat endpoint permits only one public api-version query parameter")
    return endpoint


def deployment_url(endpoint: str, model: str, api_version: str) -> str:
    """Build the exact deployment endpoint used by the supplied company example."""
    parts = urlsplit(endpoint)
    if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment):
        raise ValueError("Company endpoint must be an HTTPS base URL without query")
    if not isinstance(model, str) or not model.strip() or not isinstance(api_version, str) or not api_version.strip():
        raise ValueError("Company model and api_version must be explicit")
    return (endpoint.rstrip("/") + "/openai/deployments/" + quote(model, safe="")
            + "/chat/completions?api-version=" + quote(api_version, safe=""))


def chat_payload(model: str, messages: list, *, max_tokens: int, temperature=0.0,
                 token_parameter="max_tokens", include_model=True) -> dict:
    """Make a non-streaming body, including no unsupported sampling parameters."""
    if not isinstance(token_parameter, str) or token_parameter not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("token_parameter must be max_tokens or max_completion_tokens")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if type(include_model) is not bool:
        raise ValueError("include_model must be a boolean")
    if temperature is not None and (isinstance(temperature, bool)
            or not isinstance(temperature, (int, float)) or not math.isfinite(temperature)
            or not 0 <= temperature <= 2):
        raise ValueError("temperature must be null or finite in [0,2]")
    body = {"messages": messages, "stream": False, token_parameter: max_tokens}
    if include_model:
        body["model"] = model
    if temperature is not None:
        body["temperature"] = temperature
    return body


def auth_headers(key: str, auth: str = "bearer") -> dict:
    if auth not in {"api-key", "bearer"}:
        raise ValueError("auth must be api-key or bearer")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["api-key" if auth == "api-key" else "Authorization"] = key if auth == "api-key" else "Bearer " + key
    return headers


def requests_verify() -> str | bool:
    """Honor deployment CA settings consistently with the provided urllib sample.

    Explicit Requests variables have priority. SSL_CERT_FILE/DIR are the standard
    Python TLS settings but Requests otherwise ignores them. Verification is
    always enabled; this function never downloads or trusts a new certificate.
    """
    return (os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
            or os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR") or True)
