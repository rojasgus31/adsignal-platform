"""Storage and nearest-neighbour search for entity embeddings.

This is the only module that reads or writes vectors. The dimension is fixed
at the column level: pgvector will not accept a vector of another width, and a
same-width vector from a different model would search a space it does not
belong to. `upsert_embeddings` therefore checks the width itself, so the error
names the problem instead of surfacing as a driver-level failure.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from adsignal.warehouse.schema import apply_schema

# bge-small-en-v1.5 produces 384 dimensions. Changing the model means changing
# this constant AND migrating the column — the two cannot drift apart silently.
EMBEDDING_DIM = 384

_CREATE = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS entity_embeddings (
    entity_kind TEXT         NOT NULL,
    entity_id   TEXT         NOT NULL,
    text        TEXT         NOT NULL,
    vector      vector({EMBEDDING_DIM}) NOT NULL,
    PRIMARY KEY (entity_kind, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_embeddings_vector
    ON entity_embeddings USING hnsw (vector vector_cosine_ops);
"""


@dataclass(frozen=True, slots=True)
class EmbeddingRow:
    """One embedded entity, ready to store."""

    entity_kind: str
    entity_id: str
    text: str
    vector: Sequence[float]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One nearest-neighbour result. Lower distance is more similar."""

    entity_kind: str
    entity_id: str
    text: str
    distance: float


def ensure_vector_schema(dsn: str) -> None:
    """Create the extension, table and index if absent. Safe to call repeatedly.

    Serialised against every other `ensure_*_schema` helper by
    `apply_schema`; see that module for why concurrent `IF NOT EXISTS` DDL
    needs a lock.
    """
    apply_schema(dsn, _CREATE)


def upsert_embeddings(dsn: str, rows: Sequence[EmbeddingRow]) -> int:
    """Insert or replace embeddings keyed on (entity_kind, entity_id)."""
    if not rows:
        return 0

    for row in rows:
        if len(row.vector) != EMBEDDING_DIM:
            raise ValueError(
                f"embedding for {row.entity_kind}:{row.entity_id} has width "
                f"{len(row.vector)}, expected {EMBEDDING_DIM}"
            )

    ensure_vector_schema(dsn)
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        conn.cursor().executemany(
            """
            INSERT INTO entity_embeddings (entity_kind, entity_id, text, vector)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (entity_kind, entity_id) DO UPDATE SET
                text = EXCLUDED.text,
                vector = EXCLUDED.vector
            """,
            [(r.entity_kind, r.entity_id, r.text, list(r.vector)) for r in rows],
        )
        conn.commit()
    return len(rows)


def search_similar(
    dsn: str,
    query_vector: Sequence[float],
    limit: int = 10,
    kind: str | None = None,
) -> list[SearchHit]:
    """Return the `limit` nearest entities by cosine distance, nearest first."""
    if len(query_vector) != EMBEDDING_DIM:
        raise ValueError(f"query vector has width {len(query_vector)}, expected {EMBEDDING_DIM}")

    ensure_vector_schema(dsn)
    clause = "WHERE entity_kind = %s" if kind else ""
    params: list[object] = [Vector(list(query_vector))]
    if kind:
        params.append(kind)
    params.append(limit)

    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        rows = conn.execute(
            f"""
            SELECT entity_kind, entity_id, text, vector <=> %s AS distance
            FROM entity_embeddings
            {clause}
            ORDER BY distance
            LIMIT %s
            """,  # noqa: S608 - `clause` is a literal chosen above, never user input
            params,
        ).fetchall()

    return [SearchHit(kind_, id_, text, float(distance)) for kind_, id_, text, distance in rows]
