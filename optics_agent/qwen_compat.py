"""Contain the known Qwen-Agent 0.0.34 IPv4-only DNS import side effect.

Its optional image_search module replaces socket.getaddrinfo process-wide,
discarding IPv6 even when that tool is unused. This breaks reachable production
gateways on IPv6 networks. Restore only that exact wrapper, never an unrelated
resolver hook, and do not modify the installed third-party package.
"""
import socket
import sys


def restore_qwen_dns_resolver():
    """Undo the known Qwen wrapper after import; return whether it was restored.

    Preserve whichever resolver existed before Qwen (including operator hooks).
    This is a no-op for fixed/newer Qwen versions and unrelated DNS wrappers.
    Call during startup before worker threads or network requests are created.
    """
    module = sys.modules.get("qwen_agent.tools.image_search")
    wrapper = getattr(module, "_new_getaddrinfo", None)
    original = getattr(module, "_orig_getaddrinfo", None)
    if (callable(wrapper) and callable(original) and original is not wrapper
            and socket.getaddrinfo is wrapper):
        socket.getaddrinfo = original
        return True
    return False
