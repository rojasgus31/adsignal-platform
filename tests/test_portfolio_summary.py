"""The portfolio summary's prose and prompt, which are pure."""

from datetime import date

from adsignal.narrative.portfolio_summary import (
    PortfolioFacts,
    build_portfolio_prompt,
    summarize_portfolio,
)


def facts(**overrides: object) -> PortfolioFacts:
    base = dict(
        campaign_count=480,
        channel_count=5,
        first_day=date(2024, 1, 2),
        last_day=date(2025, 12, 31),
        days=730,
        spend=125_829_188.0,
        conversions=98_412,
        revenue=251_658_376.0,
        planted_days=1958,
        flagged_days=1918,
        escalated_days=204,
        top_causes=("budget_pacing", "tracking_error"),
        worst_campaign="Quarry Partners Brand Lift",
        worst_campaign_flags=31,
    )
    base.update(overrides)
    return PortfolioFacts(**base)  # type: ignore[arg-type]


class TestDerivedRatios:
    def test_cost_per_conversion_and_return_on_spend(self) -> None:
        f = facts(spend=100.0, conversions=4, revenue=250.0)
        assert f.cost_per_conversion == 25.0
        assert f.return_on_spend == 2.5

    def test_are_none_rather_than_dividing_by_zero(self) -> None:
        assert facts(conversions=0).cost_per_conversion is None
        assert facts(spend=0.0).return_on_spend is None


class TestSummarizePortfolio:
    def test_leads_with_the_whole_book_not_one_day(self) -> None:
        # The defect this module exists for: the dashboard's opening sentence
        # described a single day and two anomalies while the warehouse held
        # nearly two thousand flagged campaign-days.
        text = summarize_portfolio(facts())
        assert "480 campaigns" in text
        assert "5 channels" in text
        assert "1,918 campaign-days" in text

    def test_names_the_worst_campaign_rather_than_an_identifier(self) -> None:
        text = summarize_portfolio(facts())
        assert "Quarry Partners Brand Lift" in text
        assert "cmp_" not in text

    def test_omits_the_worst_campaign_when_nothing_is_flagged(self) -> None:
        text = summarize_portfolio(
            facts(flagged_days=0, escalated_days=0, top_causes=(), worst_campaign="")
        )
        assert "flagged nothing" in text
        assert "accounts for the most" not in text

    def test_says_nothing_converted_rather_than_printing_infinity(self) -> None:
        text = summarize_portfolio(facts(conversions=0))
        assert "Nothing converted" in text
        assert "inf" not in text.lower()

    def test_omits_return_on_spend_when_nothing_was_spent(self) -> None:
        assert "times spend" not in summarize_portfolio(facts(spend=0.0))

    def test_agrees_with_its_own_singular(self) -> None:
        assert "1 channel " in summarize_portfolio(facts(channel_count=1))

    def test_uses_no_long_dashes(self) -> None:
        text = summarize_portfolio(facts())
        assert "—" not in text
        assert "–" not in text

    def test_is_deterministic(self) -> None:
        assert summarize_portfolio(facts()) == summarize_portfolio(facts())


class TestBuildPortfolioPrompt:
    def test_separates_flagged_from_planted(self) -> None:
        # Shown two similar counts without the distinction, a model claims the
        # detector found every planted problem.
        prompt = build_portfolio_prompt(facts(flagged_days=1918, planted_days=1958))
        assert "flagged as unusual: 1918" in prompt
        assert "1958" not in prompt

    def test_carries_the_money_and_the_ratios(self) -> None:
        prompt = build_portfolio_prompt(facts(spend=100.0, revenue=250.0, conversions=4))
        assert "$100" in prompt
        assert "$250" in prompt
        assert "2.50 times spend" in prompt

    def test_humanizes_causes(self) -> None:
        prompt = build_portfolio_prompt(facts(top_causes=("budget_pacing",)))
        assert "budget pacing" in prompt
        assert "budget_pacing" not in prompt

    def test_says_none_rather_than_leaving_a_line_dangling(self) -> None:
        prompt = build_portfolio_prompt(facts(top_causes=(), worst_campaign=""))
        assert "none recorded" in prompt
        assert "most flagged days: none" in prompt

    def test_forbids_inventing_numbers_and_dashes(self) -> None:
        prompt = build_portfolio_prompt(facts())
        assert "Do not invent numbers" in prompt
        assert "do not use dashes" in prompt
