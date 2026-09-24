"""Chunked Parquet writing.

Rows arrive as a stream and are flushed in fixed-size row groups. Peak
memory is bounded by chunk_size, not by the total row count.
"""

from collections.abc import Iterable
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from adsignal.generator.models import AnomalyLabel, EventRow

EVENT_SCHEMA = pa.schema(
    [
        pa.field("event_date", pa.date32(), nullable=False),
        pa.field("campaign_id", pa.string(), nullable=False),
        pa.field("creative_id", pa.string(), nullable=False),
        pa.field("channel", pa.string(), nullable=False),
        pa.field("impressions", pa.int64(), nullable=False),
        pa.field("clicks", pa.int64(), nullable=False),
        pa.field("conversions", pa.int64(), nullable=False),
        pa.field("spend", pa.float64(), nullable=False),
        pa.field("revenue", pa.float64(), nullable=False),
    ]
)

LABEL_SCHEMA = pa.schema(
    [
        pa.field("event_date", pa.date32(), nullable=False),
        pa.field("campaign_id", pa.string(), nullable=False),
        pa.field("cause", pa.string(), nullable=False),
        pa.field("severity", pa.int32(), nullable=False),
    ]
)


def write_events_parquet(
    rows: Iterable[EventRow],
    destination: Path,
    chunk_size: int = 250_000,
) -> int:
    """Write event rows to Parquet in chunks. Returns the number written."""
    return _write_chunked(
        records=(_event_to_dict(row) for row in rows),
        schema=EVENT_SCHEMA,
        destination=destination,
        chunk_size=chunk_size,
    )


def write_labels_parquet(labels: Iterable[AnomalyLabel], destination: Path) -> int:
    """Write anomaly labels to Parquet. Returns the number written."""
    return _write_chunked(
        records=(_label_to_dict(label) for label in labels),
        schema=LABEL_SCHEMA,
        destination=destination,
        chunk_size=100_000,
    )


def _write_chunked(
    *,
    records: Iterable[dict[str, object]],
    schema: pa.Schema,
    destination: Path,
    chunk_size: int,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(records)
    total = 0

    with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        while True:
            chunk = list(islice(iterator, chunk_size))
            if not chunk:
                break
            writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
            total += len(chunk)

    return total


def _event_to_dict(row: EventRow) -> dict[str, object]:
    return {
        "event_date": row.event_date,
        "campaign_id": row.campaign_id,
        "creative_id": row.creative_id,
        "channel": row.channel,
        "impressions": row.impressions,
        "clicks": row.clicks,
        "conversions": row.conversions,
        "spend": row.spend,
        "revenue": row.revenue,
    }


def _label_to_dict(label: AnomalyLabel) -> dict[str, object]:
    return {
        "event_date": label.event_date,
        "campaign_id": label.campaign_id,
        "cause": label.cause,
        "severity": label.severity,
    }
