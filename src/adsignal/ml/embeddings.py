"""Text embeddings via fastembed.

`bge-small-en-v1.5` quantised to int8 ONNX: roughly 90 MB, CPU-only, no
PyTorch. That matters because the whole project runs on free CPU hosting and
an 8 GB development machine.

This module knows nothing about storage. It turns strings into vectors.
"""

from collections.abc import Sequence
from functools import cached_property

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Must equal the width of the pgvector column. See warehouse/vector_store.py.
EMBEDDING_DIM = 384


class EmbeddingModel:
    """Lazily-loaded sentence embedder.

    The model is loaded on first use rather than at construction, so importing
    this module — which the API does at startup — costs nothing.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name

    @cached_property
    def _model(self) -> TextEmbedding:
        return TextEmbedding(model_name=self._model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch. Returns one vector per input, in input order."""
        if not texts:
            return []
        for text in texts:
            _require_non_empty(text)
        return [list(map(float, vector)) for vector in self._model.embed(list(texts))]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string."""
        _require_non_empty(text)
        return self.embed([text])[0]


def _require_non_empty(text: str) -> None:
    if not text.strip():
        raise ValueError("cannot embed an empty or whitespace-only string")
