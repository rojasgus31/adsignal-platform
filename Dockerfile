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

# Bake the embedding weights (BAAI/bge-small-en-v1.5, ~90 MB) into the image.
# Nothing else fetches them at build time, so without this the first /search
# on a fresh deployment downloads them inside the request - a long hang, or a
# 500 on a host with restricted egress - and every container restart repeats
# it. Both cache variables are set because fastembed resolves its ONNX cache
# from FASTEMBED_CACHE_PATH (defaulting to a temp directory that does not
# survive a restart) and huggingface_hub resolves its own from HF_HOME. Both
# live under /app so the chown below covers them for the non-root user.
ENV HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

# Driven through EmbeddingModel rather than a hardcoded repo id, so the image
# can never cache a different model than the code loads.
RUN uv run python -c "from adsignal.ml.embeddings import EmbeddingModel; EmbeddingModel().embed_one('warm the model cache')"

RUN groupadd --gid 1000 adsignal \
    && useradd --uid 1000 --gid adsignal --create-home --shell /bin/bash adsignal \
    && mkdir -p /app/.dagster /app/data /app/.cache \
    && chown -R adsignal:adsignal /app

USER adsignal

EXPOSE 3000

CMD ["dagster", "dev", "-m", "adsignal.pipeline.definitions", "-h", "0.0.0.0", "-p", "3000"]
