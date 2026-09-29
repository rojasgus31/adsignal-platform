"""Gemini as a question-to-SQL backend.

The sibling of `GeminiProvider`: same endpoint, same transport, same
redaction (all of it in `gemini.generate_text`), a different prompt and a
different contract. It implements `SqlProvider`, not `NarrativeProvider` —
see `narrative.sqlgen` for why those are two contracts rather than one.

`model` has no default here for the same reason it has none there: the name
comes from `Settings.gemini_model`, so the two providers cannot drift onto
different models and neither carries a stale literal.
"""

import httpx

from adsignal.narrative.providers.gemini import generate_text
from adsignal.narrative.sqlgen import SqlQuestion, build_sql_prompt, extract_sql


class GeminiSqlProvider:
    """Question-to-SQL via Gemini Flash."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = "gemini"
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = client

    def generate_sql(self, request: SqlQuestion) -> str:
        """Return the model's SQL, unfenced, raising with the key redacted on failure."""
        answer = generate_text(
            build_sql_prompt(request),
            api_key=self._api_key,
            model=self._model,
            timeout=self._timeout,
            client=self._client,
        )
        return extract_sql(answer)
