"""Load the DuckDB marts into a target Postgres, for the nightly scheduler.

The pipeline runs on a GitHub Actions runner, which is ephemeral: the DuckDB
file it builds vanishes with the job. This is the step that makes the result
outlive the runner.

Loading is replace-not-append. The nightly job runs against a database that
already holds yesterday's rows, and a mart is a snapshot of the whole window,
not a delta.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from adsignal.warehouse.loader import load_mart_to_postgres
from adsignal.warehouse.pool import close_all_pools


@dataclass(frozen=True)
class LoadReport:
    """What one load did. Carries counts and timing — never the DSN."""

    tables: dict[str, int]
    duration_seconds: float


def load_all(duckdb_path: Path, target_dsn: str) -> LoadReport:
    """Copy every mart from `duckdb_path` into `target_dsn`, replacing what is there.

    `warehouse.loader` currently exposes one mart loader,
    `load_mart_to_postgres`, which loads `fct_campaign_daily` and returns its
    row count. `load_all` is the seam for more marts later: adding one means
    adding another `warehouse.loader` call and dict entry here, not changing
    this function's shape or its caller.
    """
    if not Path(duckdb_path).exists():
        raise FileNotFoundError(f"no DuckDB file at {duckdb_path}")

    started = time.monotonic()
    row_count = load_mart_to_postgres(Path(duckdb_path), target_dsn)
    tables = {"fct_campaign_daily": row_count}
    return LoadReport(tables=tables, duration_seconds=time.monotonic() - started)


def main(argv: list[str] | None = None) -> int:
    """`python -m adsignal.cli.load_marts --duckdb data/adsignal.duckdb`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb", type=Path, required=True)
    args = parser.parse_args(argv)

    from adsignal.config import get_settings

    try:
        report = load_all(args.duckdb, get_settings().postgres_dsn)
        for table, count in sorted(report.tables.items()):
            print(f"{table:24} {count:>8} rows")
        print(f"loaded in {report.duration_seconds:.1f}s")
        return 0
    finally:
        # This process exists to run one load and exit; nothing else in it
        # will reuse the pool, so there is nothing to gain by keeping it open
        # and a leaked connection to lose by not closing it.
        close_all_pools()


if __name__ == "__main__":
    sys.exit(main())
