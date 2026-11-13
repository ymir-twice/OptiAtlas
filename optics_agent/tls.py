"""Verified TLS using already-installed or explicitly configured trust roots.

This module never fetches certificates, disables verification, or trusts a
certificate merely because a server returned it.
"""
import os
import ssl
from pathlib import Path


def create_ssl_context():
    """Honor deployment CA overrides; otherwise add certifi to Python defaults.

    Requests-style bundle overrides have priority, matching our image client.
    Python's SSL_CERT_FILE and SSL_CERT_DIR can be supplied together. Invalid
    explicit settings fail closed instead of silently using a different CA set.
    """
    selected = next((name for name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
                     if os.environ.get(name)), None)
    configured = {}
    if selected:
        path = Path(os.environ[selected])
        if path.is_file():
            configured["cafile"] = str(path)
        elif path.is_dir():
            configured["capath"] = str(path)
        else:
            raise ValueError("TLS CA path is missing or invalid: " + selected)
    else:
        for name, argument, kind in (("SSL_CERT_FILE", "cafile", "is_file"),
                                     ("SSL_CERT_DIR", "capath", "is_dir")):
            if os.environ.get(name):
                path = Path(os.environ[name])
                if not getattr(path, kind)():
                    raise ValueError("TLS CA path is missing or invalid: " + name)
                configured[argument] = str(path)
    try:
        context = ssl.create_default_context(**configured)
        if not configured:
            try:
                import certifi
            except ImportError:
                pass
            else:
                context.load_verify_locations(cafile=certifi.where())
    except (OSError, ValueError):
        # A raw OpenSSL exception can contain deployment-specific file paths.
        raise ValueError("Cannot load configured or installed TLS CA certificates") from None
    return context
