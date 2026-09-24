FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first so the layer caches across source edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY dbt ./dbt
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    DAGSTER_HOME=/app/.dagster

RUN mkdir -p /app/.dagster /app/data

EXPOSE 3000

CMD ["dagster", "dev", "-m", "adsignal.pipeline.definitions", "-h", "0.0.0.0", "-p", "3000"]
