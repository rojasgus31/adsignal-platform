"""Reading embeddable text out of the DuckDB staging layer.

Campaign names and creative headlines are the only natural-language text this
project has. They live in DuckDB, not in the Postgres mart — the mart is
aggregated numbers, and widening it with text nothing else queries would break
the hot/warm split the platform is built on.
"""

from dataclasses import dataclass
from pathlib import Path

from adsignal.warehouse.duckdb_io import open_duckdb


@dataclass(frozen=True, slots=True)
class EntityText:
    """One embeddable entity: what it is, which one, and the text to embed."""

    entity_kind: str
    entity_id: str
    text: str


def read_entity_text(duckdb_path: Path) -> list[EntityText]:
    """Return campaign and creative text from the staging models."""
    with open_duckdb(duckdb_path) as conn:
        campaigns = conn.execute(
            """
            SELECT campaign_id, campaign_name, channel
            FROM main_staging.stg_campaigns
            """
        ).fetchall()
        creatives = conn.execute(
            """
            SELECT creative_id, headline, format
            FROM main_staging.stg_creatives
            """
        ).fetchall()

    rows = [
        EntityText("campaign", campaign_id, f"{name} ({channel} campaign)")
        for campaign_id, name, channel in campaigns
    ]
    rows.extend(
        EntityText("creative", creative_id, f"{headline} ({fmt} creative)")
        for creative_id, headline, fmt in creatives
    )
    return rows
