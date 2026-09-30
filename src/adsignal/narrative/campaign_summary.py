"""Writing a summary of one campaign's whole run.

The stored daily narrative answers "what happened yesterday, everywhere".
This answers "how has this campaign done", which is the question a person
looking at one campaign actually has and the one the dashboard could not
answer: its only prose described a single day across every campaign, so a
reader who had selected a campaign was reading about something else.

Two paths, same shape as the daily narrative:
- a provider writes it, when a key is configured
- `summarize_campaign` writes it locally otherwise

Both are built from `CampaignFacts` and nothing else, so no raw row and no
free text out of the database can reach a provider.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class CampaignFacts:
    """Everything a campaign summary may be written from, and nothing else.

    A closed set for the same reason `NarrativeRequest` is one: the prompt is
    built from this, so no raw row and no free text out of the database can
    reach a provider.

    It lives here rather than beside the query that fills it because the
    layering rule points this way: `warehouse` may import `narrative`, and
    `narrative` may not import `warehouse`. The prompt owns its own input
    shape, and the read builds one.
    """

    campaign_id: str
    campaign_name: str
    channel: str
    first_day: date
    last_day: date
    days: int
    spend: float
    conversions: int
    revenue: float
    planted_days: int
    """Ground truth: days the generator planted a problem on."""
    flagged_days: int
    """Model output: days the detector flagged. Not the same number."""
    escalated_days: int
    last_flagged: date | None
    top_causes: tuple[str, ...]

    @property
    def cost_per_conversion(self) -> float | None:
        """`None` rather than infinity for a campaign that converted nobody."""
        return self.spend / self.conversions if self.conversions > 0 else None


def money(value: float) -> str:
    """Whole dollars with separators. Prompts and prose share one format.

    Public because `portfolio_summary` writes the same figures the same way,
    and two summaries that format spend differently read as two systems.
    """
    return f"${value:,.0f}"


def humanize_cause(cause: str) -> str:
    """`budget_pacing` -> `budget pacing`. Shared for the reason `money` is."""
    return cause.replace("_", " ")


def build_campaign_prompt(facts: CampaignFacts) -> str:
    """Return the prompt describing one campaign's run.

    States the flagged-versus-planted distinction explicitly. Without it a
    model shown two similar counts reliably conflates them, and the sentence
    that comes back claims the detector found every planted problem.
    """
    causes = ", ".join(humanize_cause(cause) for cause in facts.top_causes) or "none recorded"
    cpa = facts.cost_per_conversion
    cpa_line = money(cpa) if cpa is not None else "not applicable, no conversions"
    return (
        "You are summarising one advertising campaign's performance for a "
        "marketing executive. Write two or three plain sentences. Say how the "
        "campaign ran overall and what deserves attention. Do not invent "
        "numbers beyond those given, and do not use dashes.\n\n"
        f"Campaign: {facts.campaign_name}\n"
        f"Channel: {facts.channel}\n"
        f"Ran from {facts.first_day.isoformat()} to {facts.last_day.isoformat()} "
        f"({facts.days} days with figures)\n"
        f"Total spend: {money(facts.spend)}\n"
        f"Total conversions: {facts.conversions}\n"
        f"Cost per conversion: {cpa_line}\n"
        f"Days the detector flagged as unusual: {facts.flagged_days}\n"
        f"Of those, severe enough to need a person: {facts.escalated_days}\n"
        f"Most common causes among flagged days: {causes}\n"
    )


def summarize_campaign(facts: CampaignFacts) -> str:
    """Compose the summary locally, from the facts alone.

    The terminal fallback, and the reason a deployment with no key still
    renders sentences. Pure: no I/O, no clock, no randomness, so the same
    facts always produce the same bytes.
    """
    sentences: list[str] = []

    span = f"over {facts.days} day{'' if facts.days == 1 else 's'}"
    sentences.append(
        f"{facts.campaign_name} ran on {facts.channel} {span}, "
        f"spending {money(facts.spend)} for {facts.conversions:,} "
        f"conversion{'' if facts.conversions == 1 else 's'}."
    )

    cpa = facts.cost_per_conversion
    if cpa is not None:
        sentences.append(f"That is {money(cpa)} per conversion.")

    if facts.flagged_days == 0:
        sentences.append("The detector flagged none of its days as unusual.")
    else:
        days = f"{facts.flagged_days} day{'' if facts.flagged_days == 1 else 's'}"
        causes = ", ".join(humanize_cause(cause) for cause in facts.top_causes)
        tail = f" The most common causes were {causes}." if causes else ""
        sentences.append(f"The detector flagged {days} as unusual.{tail}")

    if facts.escalated_days > 0:
        count = facts.escalated_days
        sentences.append(
            f"{count} of those {'was' if count == 1 else 'were'} severe enough to need a person."
        )
    elif facts.flagged_days > 0:
        sentences.append("None of them met the threshold for escalation.")

    return " ".join(sentences)
