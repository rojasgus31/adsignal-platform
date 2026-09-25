"""A written summary of one day, computed from the day's own figures.

This is the chain's terminal fallback. It exists so a deployment with no API
key still renders a readable page rather than a placeholder — which is what it
used to emit. Because it is deterministic and local, it also never fails, never
costs anything, and never says something the numbers do not support.

Pure by construction: no I/O, no clock, no randomness. The same request always
produces the same bytes.
"""

from adsignal.narrative.types import NarrativeRequest

_MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _humanize(cause: str) -> str:
    """`creative_fatigue` -> `creative fatigue`."""
    return cause.replace("_", " ")


def _plural(count: int, singular: str, plural: str) -> str:
    """Pick `singular` for exactly one, `plural` otherwise."""
    return singular if count == 1 else plural


def _format_date(request: NarrativeRequest) -> str:
    """Render `request.as_of` as `31 December 2025` without relying on `%-d`.

    `%-d` (no zero-padded day) is glibc/BSD-specific and unsupported on
    Windows. Building the string from `.day` and a fixed month-name table
    keeps the output identical across platforms.
    """
    as_of = request.as_of
    return f"{as_of.day} {_MONTH_NAMES[as_of.month - 1]} {as_of.year}"


def _describe_no_activity(request: NarrativeRequest) -> str:
    """The day has no campaign rows at all."""
    return (
        f"No campaign activity was recorded for {_format_date(request)}. "
        "The warehouse holds no rows for this date, which usually means the "
        "pipeline has not yet run for it."
    )


def _describe_anomalies(request: NarrativeRequest) -> list[str]:
    """Sentences covering anomaly volume and causes, or a clean-day note."""
    if request.anomaly_count == 0:
        return ["The detector flagged nothing unusual."]

    day_word = _plural(request.anomaly_count, "campaign-day", "campaign-days")
    parts = [f"The detector flagged {request.anomaly_count:,} {day_word}."]
    if request.top_causes:
        causes = tuple(_humanize(cause) for cause in request.top_causes[:3])
        cause_word = _plural(len(causes), "cause was", "causes were")
        parts.append(f"The most common {cause_word} {', '.join(causes)}.")
    return parts


def _describe_escalations(request: NarrativeRequest) -> str | None:
    """Sentence covering how many anomalies were severe enough to escalate.

    Escalation is only meaningful relative to anomalies that were actually
    flagged: with none flagged, `None` is returned so the paragraph does not
    claim escalations "of those" when there was no "those" to begin with.
    """
    if request.anomaly_count == 0:
        return None
    if request.escalated_count:
        verb = _plural(request.escalated_count, "was", "were")
        return f"{request.escalated_count:,} of those {verb} severe enough to escalate."
    return "None met the escalation threshold."


def summarize(request: NarrativeRequest) -> str:
    """One paragraph describing `request`."""
    if request.campaign_count == 0:
        return _describe_no_activity(request)

    campaign_word = _plural(request.campaign_count, "campaign was", "campaigns were")
    parts = [f"On {_format_date(request)}, {request.campaign_count:,} {campaign_word} active."]
    parts.extend(_describe_anomalies(request))

    escalation_sentence = _describe_escalations(request)
    if escalation_sentence is not None:
        parts.append(escalation_sentence)

    return " ".join(parts)
