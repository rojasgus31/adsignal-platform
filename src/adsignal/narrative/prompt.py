"""Turning a NarrativeRequest into a prompt.

One function, so every provider sends the same words and the benchmark
compares backends rather than prompts.
"""

from adsignal.narrative.types import NarrativeRequest


def build_prompt(request: NarrativeRequest) -> str:
    """Return the prompt describing one day's campaign health."""
    causes = ", ".join(request.top_causes) if request.top_causes else "none"
    return (
        "You are summarising one day of advertising campaign performance for a "
        "marketing operations dashboard. Write two or three plain sentences. "
        "State what happened and what deserves attention. Do not invent numbers "
        "beyond those given.\n\n"
        f"Date: {request.as_of.isoformat()}\n"
        f"Active campaigns: {request.campaign_count}\n"
        f"Anomalies detected: {request.anomaly_count}\n"
        f"Most common causes: {causes}\n"
        f"Escalated for human review: {request.escalated_count}\n"
    )
