from datetime import date

import pyarrow.parquet as pq

from adsignal.generator.models import AnomalyLabel, EventRow
from adsignal.warehouse.writer import write_events_parquet, write_labels_parquet


def _row(day: int) -> EventRow:
    return EventRow(
        event_date=date(2024, 1, day),
        campaign_id="cmp_0000_000",
        creative_id="crv_0000_000_00",
        channel="search",
        impressions=1000 + day,
        clicks=20,
        conversions=2,
        spend=15.5,
        revenue=60.25,
    )


def test_writes_all_rows_and_returns_the_count(tmp_path) -> None:
    destination = tmp_path / "events.parquet"
    written = write_events_parquet((_row(day) for day in range(1, 11)), destination)
    assert written == 10

    table = pq.read_table(destination)
    assert table.num_rows == 10
    assert table.column("impressions").to_pylist()[0] == 1001


def test_chunking_produces_one_file_with_every_row(tmp_path) -> None:
    destination = tmp_path / "events.parquet"
    written = write_events_parquet(
        (_row((index % 28) + 1) for index in range(1000)),
        destination,
        chunk_size=64,
    )
    assert written == 1000
    assert pq.read_table(destination).num_rows == 1000


def test_writer_streams_without_materializing(tmp_path) -> None:
    """Review Focus 1: the writer must never hold more than one chunk.

    The source yields more rows than the chunk size and asserts that the
    writer has flushed before it is asked for the final row.
    """
    destination = tmp_path / "events.parquet"
    high_water_mark = 0
    live_rows = 0

    def source():
        nonlocal high_water_mark, live_rows
        for index in range(500):
            live_rows += 1
            high_water_mark = max(high_water_mark, live_rows)
            yield _row((index % 28) + 1)
            live_rows -= 1

    written = write_events_parquet(source(), destination, chunk_size=50)
    assert written == 500
    assert high_water_mark == 1, "the writer pulled more than one row at a time"

    metadata = pq.ParquetFile(destination).metadata
    assert metadata.num_row_groups >= 10, "rows were not flushed in chunks"


def test_empty_source_writes_a_valid_empty_file(tmp_path) -> None:
    destination = tmp_path / "events.parquet"
    assert write_events_parquet(iter([]), destination) == 0
    assert pq.read_table(destination).num_rows == 0


def test_writes_labels(tmp_path) -> None:
    destination = tmp_path / "labels.parquet"
    labels = [
        AnomalyLabel(
            event_date=date(2024, 5, 4),
            campaign_id="cmp_0000_000",
            cause="creative_fatigue",
            severity=3,
        )
    ]
    assert write_labels_parquet(labels, destination) == 1
    table = pq.read_table(destination)
    assert table.column("cause").to_pylist() == ["creative_fatigue"]
