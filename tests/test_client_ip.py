"""Review Focus 5: behind a proxy, every visitor looks like one caller."""

from dataclasses import dataclass

from adsignal.api.client_ip import client_key


@dataclass
class _FakeClient:
    host: str


class _FakeRequest:
    """Enough of `Request` for `client_key`: a peer address and headers."""

    def __init__(self, peer: str | None, headers: dict[str, str] | None = None) -> None:
        self.client = _FakeClient(peer) if peer is not None else None
        self.headers = headers or {}


def test_with_no_proxy_the_socket_peer_is_the_client() -> None:
    request = _FakeRequest("203.0.113.7")
    assert client_key(request, trusted_hops=0) == "203.0.113.7"


def test_with_no_proxy_a_forwarded_header_is_ignored() -> None:
    """Trusting the header with no proxy in front lets anyone forge an address."""
    request = _FakeRequest("203.0.113.7", {"x-forwarded-for": "1.2.3.4"})
    assert client_key(request, trusted_hops=0) == "203.0.113.7"


def test_behind_one_proxy_the_last_forwarded_entry_is_the_client() -> None:
    """Render appends the peer it saw, so with one hop the caller is the last entry."""
    request = _FakeRequest("10.0.0.1", {"x-forwarded-for": "198.51.100.9"})
    assert client_key(request, trusted_hops=1) == "198.51.100.9"


def test_a_spoofed_prefix_cannot_displace_the_real_client() -> None:
    """A caller sending their own X-Forwarded-For prepends to the list. With one
    trusted hop we take the last entry, which is the address the proxy observed."""
    request = _FakeRequest("10.0.0.1", {"x-forwarded-for": "1.2.3.4, 198.51.100.9"})
    assert client_key(request, trusted_hops=1) == "198.51.100.9"


def test_a_header_shorter_than_the_trusted_hops_falls_back_to_the_peer() -> None:
    """A misconfiguration must not hand every caller the same forged key."""
    request = _FakeRequest("10.0.0.1", {"x-forwarded-for": "1.2.3.4"})
    assert client_key(request, trusted_hops=3) == "10.0.0.1"


def test_a_missing_header_behind_a_proxy_falls_back_to_the_peer() -> None:
    request = _FakeRequest("10.0.0.1")
    assert client_key(request, trusted_hops=1) == "10.0.0.1"


def test_a_request_with_no_peer_at_all_gets_a_stable_fallback() -> None:
    assert client_key(_FakeRequest(None), trusted_hops=0) == "unknown"
