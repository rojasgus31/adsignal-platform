"""Report the API's resident memory after a cold start and one /search.

Run inside the container, for two independent reasons.

First, the figure is only meaningful there: it needs to include the
interpreter, onnxruntime's arenas and the loaded model, and a host-side
measurement misses all three.

Second, `peak_rss_mb` below is only correct on Linux. `ru_maxrss` is
kilobytes on Linux and *bytes* on macOS, and the conversion divides by 1024
either way. Run on a macOS host this over-reports by 1024x — a 412 MB peak
prints as 421888.0 "MB" — which is loud enough to notice, but only if you
know to look.
"""

import os
import resource
import sys

from fastapi.testclient import TestClient


def peak_rss_mb() -> float:
    """Peak resident set size in MB, assuming Linux.

    `ru_maxrss` is kilobytes on Linux, so dividing by 1024 gives MB. On macOS
    it is bytes and this returns kilobytes mislabelled as MB — see the module
    docstring. Run this inside the container.
    """
    kilobytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kilobytes / 1024


def main() -> int:
    print(f"baseline           {peak_rss_mb():7.1f} MB")

    from adsignal.api.app import create_app

    client = TestClient(create_app())
    print(f"after create_app   {peak_rss_mb():7.1f} MB")

    client.get("/health")
    print(f"after /health      {peak_rss_mb():7.1f} MB")

    response = client.post("/search", json={"query": "budget pacing", "limit": 5})
    print(f"after /search      {peak_rss_mb():7.1f} MB  (status {response.status_code})")

    peak = peak_rss_mb()
    budget = float(os.environ.get("MEMORY_BUDGET_MB", "512"))
    print(f"\npeak {peak:.1f} MB against a {budget:.0f} MB budget")
    if peak > budget * 0.85:
        print("OVER 85% OF BUDGET — see Task 4 Step 4 for the fallback", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
