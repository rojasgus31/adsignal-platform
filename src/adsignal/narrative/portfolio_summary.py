"""Writing a summary of every campaign at once.

Three kinds of prose now exist, and they answer three different questions:

- the stored nightly narrative: what happened on ONE day, across everything.
  Written by the pipeline, never changes, and is the evidence that the
  nightly job ran.
- `campaign_summary`: how has THIS campaign done, over its whole run.
- this module: how is the whole book of business doing, over its whole
  history.

The dashboard led with the first and had no third, so a reader's opening
sentence described a single day and two anomalies while the warehouse held
well over a thousand. That is accurate and it is not a summary.

The field overlap with `CampaignFacts` is deliberate duplication rather than a
shared base class. Nine of the fourteen names match, but the two differ where
it matters (a campaign has a name and a channel; a portfolio has counts of
both), and a base class holding the intersection would put indirection
between the query and the prompt to save a dozen lines of field declaration.
"""

from dataclasses import dataclass
from datetime import date

from adsignal.narrative.campaign_summary import humanize_cause, money


@dataclass(frozen=True, slots=True)
class PortfolioFacts:
    """Everything a whole-warehouse summary may be written from.

    A closed set, for the same reason `CampaignFacts` is one: the prompt is
    built from this, so nothing unexpected can reach a provider.
    """

    campaign_count: int
    channel_count: int
    first_day: date
    last_day: date
    days: int
    """Distinct calendar days with figures, not campaign-days."""
    spend: float
    conversions: int
    revenue: float
    planted_days: int
    """Ground truth: campaign-days the generator planted a problem on."""
    flagged_days: int
    """Model output: campaign-days the detector flagged. Not the same number."""
    escalated_days: int
    top_causes: tuple[str, ...]
    worst_campaign: str
    """The campaign with the most flagged days. Empty when nothing is flagged."""
    worst_campaign_flags: int

    @property
    def cost_per_conversion(self) -> float | None:
        """`None` rather than infinity when nothing converted."""
        return self.spend / self.conversions if self.conversions > 0 else None

    @property
    def return_on_spend(self) -> float | None:
        """Revenue per dollar spent, or `None` when nothing was spent."""
        return self.revenue / self.spend if self.spend > 0 else None


def build_portfolio_prompt(facts: PortfolioFacts) -> str:
    """Return the prompt describing the whole warehouse.

    Names the flagged-versus-planted distinction for the same reason the
    campaign prompt does: shown two similar counts without it, a model
    reliably claims the detector found every planted problem.
    """
    causes = ", ".join(humanize_cause(cause) for cause in facts.top_causes) or "none recorded"
    cpa = facts.cost_per_conversion
    cpa_line = money(cpa) if cpa is not None else "not applicable, no conversions"
    roas = facts.return_on_spend
    roas_line = f"{roas:.2f} times spend" if roas is not None else "not applicable, no spend"
    worst = (
        f"{facts.worst_campaign} ({facts.worst_campaign_flags} flagged days)"
        if facts.worst_campaign
        else "none"
    )
    return (
        "You are briefing a marketing executive on their whole portfolio of "
        "advertising campaigns. Write three or four plain sentences. Say how "
        "the portfolio is performing overall, what the detector has been "
        "finding, and what deserves attention. Do not invent numbers beyond "
        "those given, and do not use dashes.\n\n"
        f"Campaigns: {facts.campaign_count} across {facts.channel_count} channels\n"
        f"Covering {facts.first_day.isoformat()} to {facts.last_day.isoformat()} "
        f"({facts.days} days)\n"
        f"Total spend: {money(facts.spend)}\n"
        f"Total revenue: {money(facts.revenue)}\n"
        f"Return on spend: {roas_line}\n"
        f"Total conversions: {facts.conversions}\n"
        f"Cost per conversion: {cpa_line}\n"
        f"Campaign days the detector flagged as unusual: {facts.flagged_days}\n"
        f"Of those, severe enough to need a person: {facts.escalated_days}\n"
        f"Most common causes among flagged days: {causes}\n"
        f"Campaign with the most flagged days: {worst}\n"
    )


def summarize_portfolio(facts: PortfolioFacts) -> str:
    """Compose the portfolio summary locally, from the facts alone.

    The fallback when no key is configured, and deterministic: the same facts
    always produce the same bytes.
    """
    sentences: list[str] = []

    sentences.append(
        f"{facts.campaign_count:,} campaigns across {facts.channel_count} "
        f"channel{'' if facts.channel_count == 1 else 's'} spent "
        f"{money(facts.spend)} over {facts.days:,} days."
    )

    cpa = facts.cost_per_conversion
    if cpa is not None:
        sentences.append(f"That returned {facts.conversions:,} conversions at {money(cpa)} each.")
    else:
        sentences.append("Nothing converted in that time.")

    roas = facts.return_on_spend
    if roas is not None:
        sentences.append(f"Revenue came to {roas:.2f} times spend.")

    if facts.flagged_days == 0:
        sentences.append("The detector has flagged nothing.")
    else:
        causes = ", ".join(humanize_cause(cause) for cause in facts.top_causes)
        tail = f", most often {causes}" if causes else ""
        sentences.append(
            f"The detector flagged {facts.flagged_days:,} campaign-days as unusual{tail}."
        )
        if facts.escalated_days > 0:
            sentences.append(
                f"{facts.escalated_days:,} of those were severe enough to need a person."
            )
        if facts.worst_campaign:
            sentences.append(
                f"{facts.worst_campaign} accounts for the most, with "
                f"{facts.worst_campaign_flags:,}."
            )

    return " ".join(sentences)
