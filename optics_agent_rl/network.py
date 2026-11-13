"""Small dual-stack helpers shared by launchers; importing this module does no I/O."""
from __future__ import annotations

import ipaddress
import os
import re
import socket


def host(value, name="host", *, allow_loopback=False, allow_unspecified=False):
    """Accept a bare IPv4/IPv6 literal or DNS name, never a URL or host:port."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a bare IP address or hostname")
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        if not re.fullmatch(r"(?=.{1,253}\Z)[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?", value):
            raise ValueError(f"{name} must be a bare IPv4/IPv6 address or hostname, without a port or URL") from None
        if not allow_loopback and (value.lower() == "localhost" or value.lower().endswith(".localhost")):
            raise ValueError(f"{name} cannot be localhost")
        return value
    if "%" in value:
        raise ValueError(f"{name} must not contain an IPv6 zone identifier")
    mapped = getattr(address, "ipv4_mapped", None)
    checked = mapped or address
    if checked.is_multicast or (checked.is_loopback and not allow_loopback) or (checked.is_unspecified and not allow_unspecified):
        raise ValueError(f"{name} must be a reachable non-loopback node address")
    return str(address)


def family(ip):
    return socket.AF_INET6 if ipaddress.ip_address(ip).version == 6 else socket.AF_INET


def resolve_host(value, *, allow_loopback=False, preferred_family=None):
    value = host(value, allow_loopback=allow_loopback)
    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        pass
    if preferred_family is None:
        preferred_family = {"AF_INET6": socket.AF_INET6, "AF_INET": socket.AF_INET}.get(
            os.environ.get("ARNOLD_NCCL_SOCKET_FAMILY", ""))
    results = socket.getaddrinfo(value, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    if preferred_family:
        results.sort(key=lambda entry: entry[0] != preferred_family)
    for af, _, _, _, address in results:
        if af not in (socket.AF_INET, socket.AF_INET6):
            continue
        try:
            return host(address[0], allow_loopback=allow_loopback)
        except ValueError:
            continue
    raise ValueError(f"{value} does not resolve to a permitted node address")


def endpoint(value, port):
    value = host(value, allow_loopback=True, allow_unspecified=True)
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer in [0, 65535]")
    return f"[{value}]:{port}" if ":" in value else f"{value}:{port}"


def http_url(value, port, path=""):
    return "http://" + endpoint(value, port) + path


def socket_for(ip, kind=socket.SOCK_STREAM):
    return socket.socket(family(ip), kind)


def health_host(bind_host):
    return {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(bind_host, bind_host)
