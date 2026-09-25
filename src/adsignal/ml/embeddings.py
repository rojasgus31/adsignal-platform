"""Text embeddings via fastembed.

`bge-small-en-v1.5` quantised to int8 ONNX: roughly 90 MB, CPU-only, no
PyTorch. That matters because the whole project runs on free CPU hosting and
an 8 GB development machine.

This module knows nothing about storage. It turns strings into vectors.
"""

import threading
from collections.abc import Sequence

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Must equal the width of the pgvector column. See warehouse/vector_store.py.
EMBEDDING_DIM = 384


class EmbeddingModel:
    """Lazily-loaded sentence embedder, safe to share across threads.

    The model is loaded on first use rather than at construction, so importing
    this module — which the API does at startup — costs nothing.

    That first load is serialised. `/search` is a sync `def`, so Starlette
    dispatches it to the anyio worker threadpool where several requests run
    genuinely concurrently; an unsynchronised lazy attribute would let each
    of the first concurrent requests construct its own `TextEmbedding`,
    transiently holding several ~90 MB models inside a 1 GB container. The
    double-checked read means the lock is taken once, on the cold path, and
    never on the hot one.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._loaded: TextEmbedding | None = None
        self._load_lock = threading.Lock()

    @property
    def _model(self) -> TextEmbedding:
        """The underlying embedder, constructed exactly once."""
        if self._loaded is None:
            with self._load_lock:
                if self._loaded is None:
                    self._loaded = TextEmbedding(model_name=self._model_name)
        return self._loaded

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
