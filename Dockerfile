FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first so the layer caches across source edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY dbt ./dbt
RUN uv sync --frozen --no-dev

# The dbt project's packages (dbt_utils) are never committed or copied from
# the host (see .dockerignore) — install them at build time so the image is
# self-sufficient and does not depend on the build host having already run
# `dbt deps`.
RUN uv run dbt deps --project-dir dbt --profiles-dir dbt

ENV PATH="/app/.venv/bin:$PATH" \
    DAGSTER_HOME=/app/.dagster

RUN groupadd --gid 1000 adsignal \
    && useradd --uid 1000 --gid adsignal --create-home --shell /bin/bash adsignal \
    && mkdir -p /app/.dagster /app/data \
    && chown -R adsignal:adsignal /app

USER adsignal

EXPOSE 3000

CMD ["dagster", "dev", "-m", "adsignal.pipeline.definitions", "-h", "0.0.0.0", "-p", "3000"]
