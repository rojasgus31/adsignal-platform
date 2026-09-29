"""Deployment check: does the configured Gemini model actually answer?

`ADSIGNAL_GEMINI_MODEL` (default `config.DEFAULT_GEMINI_MODEL`) is a claim
about someone else's service, and this repository has now shipped two names
that were wrong in two different ways. `gemini-2.0-flash` was retired and
answered 404 to a valid key. `gemini-3.5-flash` was read out of a live
`ListModels` response — non-preview, `generateContent` supported — and
answered **503 to every call**, while three of its siblings answered fine.
Availability is per-model and patternless.

Every test that covered either name used a mocked transport, so both passed
the whole suite. The only check that could have caught them is the one this
module performs: send a real prompt with the real key and see what comes
back.

    python -m adsignal.cli.check_model

It takes no arguments, and it is the same shape as
`cli.bootstrap_readonly_role`: configuration comes from the environment
through `Settings`, the work is one function a test can call directly, and
`main` turns the result into an exit code and a line of text.

**It makes one billed API call.** That is the point — a free check is a check
of a list, and a list is what shipped the bug — but it is why this is a
command a person runs rather than anything the API or the suite does.

Nothing here prints, logs or returns the key. The prompt is a fixed literal,
and the only failure text it reports comes from `providers.gemini`, whose
messages replace the URL and the key with the word REDACTED by construction
(`narrative.types.ProviderUnavailable`).
"""

import sys
from dataclasses import dataclass

import httpx

from adsignal.narrative.providers.gemini import generate_text
from adsignal.narrative.types import ProviderUnavailable

# The smallest prompt that still exercises a real generation: it must produce
# tokens, because a model that accepts a request and returns an empty
# candidate list is as broken as one that answers 503 — and `generate_text`
# raises on that shape too.
_PROBE_PROMPT = "Reply with the single word: ok"

# Long enough that a slow-but-working model is not reported as broken, short
# enough that a person running this at a deploy is not left waiting. Not read
# from `narrative_timeout_seconds`: that setting bounds a request inside a
# chain with a budget, and this is one deliberate call with a person watching.
PROBE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class ModelCheck:
    """One live call's outcome. Carries no key, no URL and no prompt.

    `answered` is the whole verdict. `detail` is a sentence for a person:
    either how many characters came back, or the provider's own redacted
    failure string — `gemini returned 503 (url REDACTED, key REDACTED)` — which
    is what distinguishes a transient outage from an expired key or a retired
    model.
    """

    model: str
    answered: bool
    detail: str


def check_model(
    api_key: str,
    model: str,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    client: httpx.Client | None = None,
) -> ModelCheck:
    """Call `model` once and report whether it produced text.

    Never raises for a provider failure: a model that does not answer is the
    result this function exists to return, not an error. It does propagate a
    programming error, because that is not a fact about Google.

    `ValueError` from `generate_text` means the response arrived but had no
    candidate text in it — a model that accepts requests and generates
    nothing, which is a failure for this deployment's purposes even though
    the HTTP call succeeded.

    `client` exists so a test can exercise the branches without calling
    Google; `main` never passes one, so the command this module documents
    always makes a real request.
    """
    try:
        answer = generate_text(
            _PROBE_PROMPT, api_key=api_key, model=model, timeout=timeout, client=client
        )
    except ProviderUnavailable as unavailable:
        # Redacted by construction; see the module docstring.
        return ModelCheck(model=model, answered=False, detail=str(unavailable))
    except ValueError:
        # Deliberately not `str(error)`: `_extract_text` composes its message
        # safely, but this branch is about the response's shape and the type
        # name says all of it.
        return ModelCheck(
            model=model, answered=False, detail="the response carried no generated text"
        )
    return ModelCheck(model=model, answered=True, detail=f"answered with {len(answer)} characters")


def main() -> int:
    """`python -m adsignal.cli.check_model`. 0 when the model answered.

    Exits 2 when no key is configured — there is nothing to check, and a
    silent success would be the same false assurance as reading a list.
    Exits 1 when the key is present and the model did not answer.
    """
    from adsignal.config import get_settings

    settings = get_settings()
    key = settings.gemini_api_key.get_secret_value()
    if not key:
        print(
            "ADSIGNAL_GEMINI_API_KEY is not set; there is no key to check the model with",
            file=sys.stderr,
        )
        return 2

    result = check_model(key, settings.gemini_model)
    if not result.answered:
        print(f"model {result.model} did not answer: {result.detail}", file=sys.stderr)
        return 1
    print(f"model {result.model} answered ({result.detail})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
