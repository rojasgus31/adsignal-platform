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

    parts = [f"The detector flagged {request.anomaly_count:,} campaign-days."]
    if request.top_causes:
        causes = ", ".join(_humanize(cause) for cause in request.top_causes[:3])
        parts.append(f"The most common causes were {causes}.")
    return parts


def _describe_escalations(request: NarrativeRequest) -> str:
    """Sentence covering how many anomalies were severe enough to escalate."""
    if request.escalated_count:
        return f"{request.escalated_count:,} of those were severe enough to escalate."
    return "None met the escalation threshold."


def summarize(request: NarrativeRequest) -> str:
    """One paragraph describing `request`."""
    if request.campaign_count == 0:
        return _describe_no_activity(request)

    parts = [f"On {_format_date(request)}, {request.campaign_count:,} campaigns were active."]
    parts.extend(_describe_anomalies(request))
    parts.append(_describe_escalations(request))

    return " ".join(parts)
