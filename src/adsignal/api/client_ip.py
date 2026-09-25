"""Which address to treat as the caller's, given how many proxies are in front.

`request.client.host` is the socket peer. Deployed behind Render's edge that is
the same address for every visitor, so a per-IP rate limit becomes one global
budget and a single caller locks out the rest.

`X-Forwarded-For` carries the chain, but a caller may send their own value,
which arrives *prepended* to whatever the proxies append. Counting from the
right by the number of proxies actually in front is what makes the value
trustworthy: entries to the left of that position are caller-supplied and are
never read.

HTTP allows a header field to repeat as multiple lines that are semantically
one comma-joined value, in order. Reading only the first line (as a naive
`headers.get` does) lets an edge that emits its own `X-Forwarded-For` line,
rather than appending to the caller's, hand the caller full control of the
value this module trusts — so every line is collected with `getlist` and
joined before parsing, the same fix uvicorn's own `ProxyHeadersMiddleware`
applies.
"""

import ipaddress
import logging
from functools import cache
from typing import Protocol

FALLBACK_KEY = "unknown"

_logger = logging.getLogger(__name__)


class _HasHost(Protocol):
    """The part of `starlette.datastructures.Address` this module needs."""

    @property
    def host(self) -> str: ...


class _HasHeaders(Protocol):
    """The part of `starlette.datastructures.Headers` this module needs."""

    def getlist(self, key: str) -> list[str]: ...


class _HasClientAndHeaders(Protocol):
    """The part of `fastapi.Request` this module needs."""

    @property
    def client(self) -> _HasHost | None: ...

    @property
    def headers(self) -> _HasHeaders: ...


@cache
def _warn_short_chain_once(trusted_hops: int) -> None:
    """Log once per configured hop count that the short-chain fallback fired.

    Behind a proxy the socket peer is the same address for every visitor, so
    this fallback collapses every caller onto one rate-limit key — a global
    denial of service reintroduced by a `trusted_proxy_hops` set higher than
    the real chain. `cache` makes this fire once per distinct hop count
    rather than once per request, so a burst of misconfigured traffic logs
    one line, not thousands. No header contents or secrets are logged.
    """
    _logger.warning(
        "client_key: X-Forwarded-For had fewer entries than "
        "trusted_proxy_hops=%d configured; falling back to the socket peer, "
        "which is the same address for every caller behind a proxy. Every "
        "request is about to share one rate-limit budget — check the "
        "deployment's trusted_proxy_hops against the real number of proxies.",
        trusted_hops,
    )


def _normalize_address(value: str) -> str | None:
    """Parse `value` as an IP address, or `None` if it is not one.

    Strips a bracketed IPv6 host (`[::1]:41234` -> `::1`) or a trailing IPv4
    port (`198.51.100.1:41234` -> `198.51.100.1`) before parsing, and
    canonicalizes the result (`2001:0db8::0001` -> `2001:db8::1`) so the same
    caller under different spellings, or with a per-connection ephemeral port
    appended by some proxy in the chain, still lands on the same key. A value
    that is not an address at all — including the literal `"unknown"`, which
    some proxies emit for an unresolvable peer — returns `None` so the caller
    can fall back to the socket peer instead of minting a fresh or colliding
    key.
    """
    candidate = value
    if candidate.startswith("["):
        end = candidate.find("]")
        if end != -1:
            candidate = candidate[1:end]
    elif candidate.count(":") == 1:
        host, _, port = candidate.rpartition(":")
        if port.isdigit():
            candidate = host
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


def client_key(request: _HasClientAndHeaders, trusted_hops: int) -> str:
    """The caller's address, or `"unknown"` when there is nothing to key on."""
    peer = request.client.host if request.client is not None else None
    if trusted_hops <= 0:
        return peer or FALLBACK_KEY

    forwarded = ", ".join(request.headers.getlist("x-forwarded-for"))
    entries = [part.strip() for part in forwarded.split(",") if part.strip()]
    if len(entries) < trusted_hops:
        # Fewer entries than proxies means the chain is not what we were told.
        # Falling back to the peer over-throttles; trusting it under-throttles.
        _warn_short_chain_once(trusted_hops)
        return peer or FALLBACK_KEY

    normalized = _normalize_address(entries[-trusted_hops])
    if normalized is None:
        return peer or FALLBACK_KEY
    return normalized
