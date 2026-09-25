"""Review Focus 5: behind a proxy, every visitor looks like one caller."""

from dataclasses import dataclass

from starlette.datastructures import Headers

from adsignal.api.client_ip import client_key


@dataclass
class _FakeClient:
    host: str


class _FakeRequest:
    """Enough of `Request` for `client_key`: a peer address and headers.

    `headers` is a real `starlette.datastructures.Headers`, not a plain
    `dict`, so it can represent a header repeated as separate lines the way
    an ASGI server actually delivers one — a `dict` collapses repeats and so
    cannot distinguish `.get` (first line only) from `.getlist` (every line).
    """

    def __init__(self, peer: str | None, headers: list[tuple[str, str]] | None = None) -> None:
        self.client = _FakeClient(peer) if peer is not None else None
        raw = [(name.encode(), value.encode()) for name, value in headers or []]
        self.headers = Headers(raw=raw)


def test_with_no_proxy_the_socket_peer_is_the_client() -> None:
    request = _FakeRequest("203.0.113.7")
    assert client_key(request, trusted_hops=0) == "203.0.113.7"


def test_with_no_proxy_a_forwarded_header_is_ignored() -> None:
    """Trusting the header with no proxy in front lets anyone forge an address."""
    request = _FakeRequest("203.0.113.7", [("x-forwarded-for", "1.2.3.4")])
    assert client_key(request, trusted_hops=0) == "203.0.113.7"


def test_behind_one_proxy_the_last_forwarded_entry_is_the_client() -> None:
    """Render appends the peer it saw, so with one hop the caller is the last entry."""
    request = _FakeRequest("10.0.0.1", [("x-forwarded-for", "198.51.100.9")])
    assert client_key(request, trusted_hops=1) == "198.51.100.9"


def test_a_spoofed_prefix_cannot_displace_the_real_client() -> None:
    """A caller sending their own X-Forwarded-For prepends to the list. With one
    trusted hop we take the last entry, which is the address the proxy observed."""
    request = _FakeRequest("10.0.0.1", [("x-forwarded-for", "1.2.3.4, 198.51.100.9")])
    assert client_key(request, trusted_hops=1) == "198.51.100.9"


def test_a_spoofed_prefix_cannot_displace_the_real_client_with_two_trusted_hops() -> None:
    """The index arithmetic is only ever exercised at -1 otherwise; this pins -2."""
    request = _FakeRequest("10.0.0.1", [("x-forwarded-for", "1.2.3.4, 198.51.100.9, 10.0.0.2")])
    assert client_key(request, trusted_hops=2) == "198.51.100.9"


def test_a_repeated_header_line_does_not_let_the_first_line_win() -> None:
    """Finding 1a: HTTP allows X-Forwarded-For to repeat as separate lines, which are
    semantically one comma-joined value in order. Reading only the first line (as a
    naive `headers.get` does) lets an edge that emits its own line, rather than
    appending to the caller's, hand the caller full control of the key. With one
    trusted hop, the edge's line (the second one here) must be the one that wins,
    never the caller-supplied first line."""
    request = _FakeRequest(
        "10.0.0.1",
        [("x-forwarded-for", "9.9.9.9"), ("x-forwarded-for", "198.51.100.1")],
    )
    assert client_key(request, trusted_hops=1) == "198.51.100.1"


def test_a_header_shorter_than_the_trusted_hops_falls_back_to_the_peer() -> None:
    """A misconfiguration must not collapse every caller onto one shared key.

    Behind a proxy the peer is the same address for every visitor, so this
    fallback firing is itself a global denial of service on misconfiguration
    (Review Focus 5, finding 2) — over-throttling, not a forged key, but every
    bit as much a failure of the limit. `client_key` logs this once so it is
    diagnosable rather than silent.
    """
    request = _FakeRequest("10.0.0.1", [("x-forwarded-for", "1.2.3.4")])
    assert client_key(request, trusted_hops=3) == "10.0.0.1"


def test_a_missing_header_behind_a_proxy_falls_back_to_the_peer() -> None:
    request = _FakeRequest("10.0.0.1")
    assert client_key(request, trusted_hops=1) == "10.0.0.1"


def test_a_request_with_no_peer_at_all_gets_a_stable_fallback() -> None:
    assert client_key(_FakeRequest(None), trusted_hops=0) == "unknown"


def test_a_port_suffix_does_not_mint_a_new_key_per_connection() -> None:
    """Finding 1e: a proxy that appends an ephemeral port must not defeat the limiter."""
    request = _FakeRequest("10.0.0.1", [("x-forwarded-for", "198.51.100.9:41234")])
    assert client_key(request, trusted_hops=1) == "198.51.100.9"


def test_a_bracketed_ipv6_entry_normalizes_the_same_as_the_bare_form() -> None:
    bracketed = _FakeRequest("10.0.0.1", [("x-forwarded-for", "[2001:db8::1]:41234")])
    bare = _FakeRequest("10.0.0.1", [("x-forwarded-for", "2001:0db8:0000::0001")])
    assert client_key(bracketed, trusted_hops=1) == client_key(bare, trusted_hops=1)


def test_the_literal_unknown_does_not_collide_with_the_fallback_key() -> None:
    """squid emits the literal token "unknown" for an unresolvable peer. It must not
    merge that caller with every peerless request into one shared bucket."""
    request = _FakeRequest("10.0.0.1", [("x-forwarded-for", "unknown")])
    assert client_key(request, trusted_hops=1) == "10.0.0.1"
