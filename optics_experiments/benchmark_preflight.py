"""Unauthenticated connectivity checks before remote benchmark workers start."""
from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request

from optics_agent.providers import ProviderError, _NoRedirect, _endpoint, transport_error_code
from optics_agent.tls import create_ssl_context


_TIMEOUT_SECONDS = 15
_UNAUTHENTICATED_NOTE = "Connectivity only; credentials and model availability have not been checked."


def _failure(role, host, code, logger, *, error_type=None):
    if logger:
        logger.event("benchmark_preflight_failed", role=role, host=host, code=code,
                     error_type=error_type, authentication_checked=False)
    if code in {"tls_certificate_error", "tls_error", "tls_configuration_error"}:
        hint = ("Check the configured HTTPS gateway and company proxy. If a company CA is required, "
                "obtain its trusted CA bundle from your administrator and set REQUESTS_CA_BUNDLE "
                "to that existing file. Keep certificate verification enabled.")
    elif code in {"dns_error", "connection_refused", "request_timeout", "connection_reset", "transport_error"}:
        hint = "Check DNS, outbound network/proxy access, and the configured gateway, then retry."
    elif code == "redirect_not_allowed":
        hint = "Configure the final API endpoint directly; redirects are not followed."
    elif code == "invalid_endpoint":
        hint = ("Set benchmark.actor.chat_completions_url and benchmark.image.endpoint to valid HTTPS "
                "URLs without embedded credentials; only a public api-version query is supported.")
    else:
        hint = "Check the gateway service status before retrying the benchmark."
    raise RuntimeError(f"Benchmark preflight failed: role={role} host={host or 'unavailable'} "
                       f"code={code}. No workers started. {hint}") from None


def _endpoint_settings(cfg, role, field, logger):
    settings = cfg.get("benchmark", {}).get(role)
    value = settings.get(field) if isinstance(settings, dict) else None
    try:
        if not isinstance(value, str) or not value or any(char.isspace() for char in value):
            raise ValueError()
        parts = urllib.parse.urlsplit(value)
        # Force validation of malformed/non-numeric ports before creating a request.
        parts.port
        _endpoint(urllib.parse.urlunsplit(parts._replace(query="")))
        query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if len(query) > 1 or any(key != "api-version" or not item for key, item in query):
            raise ValueError()
        return value, parts.hostname
    except (TypeError, ValueError):
        _failure(role, None, "invalid_endpoint", logger)


def check_remote_endpoints(cfg, logger=None):
    """GET the two configured gateways without keys, bodies, redirects or retries.

    A 401/403/404/405 response proves that DNS, connection and TLS worked, not
    that any credential, model or POST payload has been accepted. No image or
    model generation is requested. Server errors stop launch for operator review.
    """
    fields = [("image", "endpoint")] if cfg.get("benchmark", {}).get("architecture") == "direct_api" else [
        ("actor", "chat_completions_url"), ("image", "endpoint")]
    endpoints = [_endpoint_settings(cfg, role, field, logger) for role, field in fields]
    try:
        context = create_ssl_context()
    except ValueError:
        _failure("tls", None, "tls_configuration_error", logger)
    opener = urllib.request.build_opener(_NoRedirect(), urllib.request.HTTPSHandler(context=context))
    records = []
    for (role, _), (endpoint, host) in zip(fields, endpoints):
        if logger:
            logger.event("benchmark_preflight_started", role=role, host=host,
                         timeout_seconds=_TIMEOUT_SECONDS, authentication_checked=False)
        request = urllib.request.Request(endpoint, method="GET")
        try:
            with opener.open(request, timeout=_TIMEOUT_SECONDS) as response:
                status = response.status
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
        except ProviderError as error:
            _failure(role, host, error.code, logger, error_type=error.error_type)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            _failure(role, host, transport_error_code(error), logger,
                     error_type=type(error.reason if isinstance(error, urllib.error.URLError)
                                     and isinstance(error.reason, BaseException) else error).__name__)
        except ValueError:
            _failure(role, host, "invalid_endpoint", logger)
        if 300 <= status < 400:
            _failure(role, host, "redirect_not_allowed", logger)
        if not 200 <= status < 500:
            _failure(role, host, "http_" + str(status), logger)
        record = {"role": role, "host": host, "http_status": status,
                  "authentication_checked": False, "note": _UNAUTHENTICATED_NOTE}
        records.append(record)
        if logger:
            logger.event("benchmark_preflight_succeeded", **record)
    return records
