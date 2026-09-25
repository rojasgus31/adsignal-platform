"""Assembling the provider chain from configuration.

Order is deliberate: free tiers first, the paid model last, and a stub at the
end so a deployment with no keys at all still renders a dashboard instead of
erroring.
"""

from adsignal.config import Settings
from adsignal.narrative.chain import NarrativeChain
from adsignal.narrative.providers.anthropic import AnthropicProvider
from adsignal.narrative.providers.gemini import GeminiProvider
from adsignal.narrative.providers.groq import GroqProvider
from adsignal.narrative.providers.jev import JevProvider
from adsignal.narrative.providers.stub import StubProvider
from adsignal.narrative.types import NarrativeProvider


def build_chain(settings: Settings) -> NarrativeChain:
    """Return a chain of every configured provider, stub last."""
    providers: list[NarrativeProvider] = []
    timeout = settings.narrative_timeout_seconds

    if settings.gemini_api_key:
        providers.append(GeminiProvider(settings.gemini_api_key, timeout=timeout))
    if settings.groq_api_key:
        providers.append(GroqProvider(settings.groq_api_key, timeout=timeout))
    if settings.jev_api_key:
        providers.append(JevProvider(settings.jev_api_key, timeout=timeout))
    if settings.anthropic_api_key:
        providers.append(AnthropicProvider(settings.anthropic_api_key, timeout=timeout))

    providers.append(
        StubProvider(
            "Narrative generation is not configured; this is placeholder text.",
            name="stub",
        )
    )
    return NarrativeChain(providers)
