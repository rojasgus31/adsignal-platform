"""The image must ship the embedding weights, not fetch them per deployment.

These assertions are on the build definition rather than on a built image:
building is a CI/deployment concern, but "the weights are baked in" is a
contract the suite can hold, and it is the kind of line that is silently
dropped in a future Dockerfile edit. Without it the first `/search` on a
fresh deployment downloads ~90 MB from HuggingFace inside the request.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (_REPO_ROOT / "Dockerfile").read_text()
COMPOSE = (_REPO_ROOT / "docker-compose.yml").read_text()

# fastembed reads its ONNX cache path from the first; huggingface_hub, which
# does the actual download, reads its own from the second.
_CACHE_VARIABLES = ("FASTEMBED_CACHE_PATH", "HF_HOME")


def test_the_image_sets_both_model_cache_locations() -> None:
    for variable in _CACHE_VARIABLES:
        assert f"{variable}=/app/" in DOCKERFILE, f"{variable} is unset or outside /app"


def test_the_image_prefetches_the_weights_at_build_time() -> None:
    """Through `EmbeddingModel`, so the cache can never hold another model."""
    assert "EmbeddingModel().embed_one(" in DOCKERFILE


def test_the_prefetch_runs_before_the_cache_directory_is_chowned() -> None:
    """Otherwise the non-root runtime user cannot read what the build wrote."""
    assert DOCKERFILE.index("EmbeddingModel().embed_one(") < DOCKERFILE.index("chown -R adsignal")
    assert "/app/.cache" in DOCKERFILE.split("groupadd")[1]


def test_compose_does_not_override_the_cache_locations() -> None:
    """An override in compose would send the runtime looking somewhere empty."""
    for variable in _CACHE_VARIABLES:
        assert variable not in COMPOSE
