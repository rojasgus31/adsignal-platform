"""Which address to treat as the caller's, given how many proxies are in front.

`request.client.host` is the socket peer. Deployed behind Render's edge that is
the same address for every visitor, so a per-IP rate limit becomes one global
budget and a single caller locks out the rest.

`X-Forwarded-For` carries the chain, but a caller may send their own value,
which arrives *prepended* to whatever the proxies append. Counting from the
right by the number of proxies actually in front is what makes the value
trustworthy: entries to the left of that position are caller-supplied and are
never read.
"""

from typing import Protocol

FALLBACK_KEY = "unknown"


class _HasHost(Protocol):
    """The part of `starlette.datastructures.Address` this module needs."""

    @property
    def host(self) -> str: ...


class _HasClientAndHeaders(Protocol):
    """The part of `fastapi.Request` this module needs."""

    @property
    def client(self) -> _HasHost | None: ...

    @property
    def headers(self) -> object: ...


def client_key(request: _HasClientAndHeaders, trusted_hops: int) -> str:
    """The caller's address, or `"unknown"` when there is nothing to key on."""
    peer = request.client.host if request.client is not None else None
    if trusted_hops <= 0:
        return peer or FALLBACK_KEY

    forwarded: str = request.headers.get("x-forwarded-for", "")  # type: ignore[attr-defined]
    entries = [part.strip() for part in forwarded.split(",") if part.strip()]
    if len(entries) < trusted_hops:
        # Fewer entries than proxies means the chain is not what we were told.
        # Falling back to the peer over-throttles; trusting it under-throttles.
        return peer or FALLBACK_KEY
    return entries[-trusted_hops]
