"""CI entry point for the model quality gate.

Run as `python -m adsignal.cli.quality_gate`. Exits non-zero when a training
run breaches either committed floor, so a model regression cannot reach main
unnoticed.

This module exists because the gate's *policy* lives in `adsignal.ml.gate`,
which must not import `config` or `warehouse`. The composition root is where
those two are allowed to meet: it reads the settings, pulls the mart, and
turns the policy's verdict into an exit code.
"""

import sys

from adsignal.config import get_settings
from adsignal.ml.gate import gate_failures, report_lines
from adsignal.ml.training import train_and_evaluate
from adsignal.warehouse.reader import read_campaign_daily


def main() -> int:
    """Train on the mart, evaluate on the holdout, and gate on both models."""
    settings = get_settings()
    result = train_and_evaluate(read_campaign_daily(settings.postgres_dsn), seed=settings.seed)

    for line in report_lines(result):
        print(line)

    failures = gate_failures(result)
    for failure in failures:
        print(failure)
    if failures:
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
