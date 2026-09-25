"""Report the API's resident memory after a cold start, one /search and one /score.

Run inside the container, for two independent reasons.

First, the figure is only meaningful there: it needs to include the
interpreter, onnxruntime's arenas and the loaded models, and a host-side
measurement misses all three.

`/score` is called because it is the only request that unpickles the fitted
detector and triage estimators into this process. A run where it answers
anything but 200 has NOT measured them, and the peak it prints is an
understatement — so the call below prints its status and `main` refuses to
report a pass without it.

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

# `adsignal.ml.features.DEFAULT_BASELINE_WINDOW`, not imported: this script
# runs before (and independently of) the app it measures, and importing the
# package at module scope would move the baseline reading it prints first.
_MIN_HISTORY_DAYS = 7


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

    scored = _score_one_campaign_day(client)
    print(f"after /score       {peak_rss_mb():7.1f} MB  (status {scored})")

    peak = peak_rss_mb()
    budget = float(os.environ.get("MEMORY_BUDGET_MB", "512"))
    print(f"\npeak {peak:.1f} MB against a {budget:.0f} MB budget")
    if scored != 200:
        print(
            f"/score answered {scored}, not 200: the scoring models were never loaded, "
            "so this peak does not include them",
            file=sys.stderr,
        )
        return 1
    if peak > budget * 0.85:
        print("OVER 85% OF BUDGET — see Task 4 Step 4 for the fallback", file=sys.stderr)
        return 1
    return 0


def _score_one_campaign_day(client: TestClient) -> int:
    """POST /score for a campaign-day old enough to have its rolling baselines.

    The campaign list and its series come from the API itself rather than
    from a query: this script measures the deployed process, and giving it a
    second, independent way to reach the database would measure something
    else. A campaign's first days cannot be scored at all (their baselines
    are NaN), so this picks a day well inside the series.
    """
    campaigns = client.get("/campaigns").json()
    if not campaigns:
        return 0
    campaign_id = campaigns[0]["campaign_id"]
    points = client.get(f"/campaigns/{campaign_id}/metrics").json()["points"]
    if len(points) <= _MIN_HISTORY_DAYS:
        return 0
    return client.post(
        "/score",
        json={"campaign_id": campaign_id, "event_date": points[-1]["event_date"]},
    ).status_code


if __name__ == "__main__":
    raise SystemExit(main())
