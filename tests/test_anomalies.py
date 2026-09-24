import random
from collections import Counter
from datetime import date

from adsignal.generator.anomalies import AnomalyInjector
from adsignal.generator.catalog import build_catalog
from adsignal.generator.events import generate_events
from adsignal.generator.models import ANOMALY_CAUSES


def _rows(seed: int = 1):
    catalog = build_catalog(
        seed=5,
        n_advertisers=3,
        campaigns_per_advertiser=4,
        creatives_per_campaign=2,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )
    return generate_events(catalog, seed=seed)


def test_injector_preserves_row_count() -> None:
    before = len(list(_rows()))
    injector = AnomalyInjector(seed=2, rate=0.05)
    after = len(list(injector.rows(_rows())))
    assert before == after


def test_labels_are_produced_and_well_formed() -> None:
    injector = AnomalyInjector(seed=2, rate=0.05)
    list(injector.rows(_rows()))
    assert injector.labels, "no anomalies were injected at a 5% rate"
    for label in injector.labels:
        assert label.cause in ANOMALY_CAUSES
        assert 1 <= label.severity <= 5


def test_at_most_one_anomaly_per_campaign_day() -> None:
    """Review Focus 5: duplicate labels for one campaign-day would make the
    supervised training set silently ambiguous."""
    injector = AnomalyInjector(seed=3, rate=0.20)
    list(injector.rows(_rows()))
    counts = Counter((label.campaign_id, label.event_date) for label in injector.labels)
    duplicates = {key: count for key, count in counts.items() if count > 1}
    assert not duplicates, f"duplicate campaign-day labels: {duplicates}"


def test_injection_is_deterministic_for_a_fixed_input_order() -> None:
    first = AnomalyInjector(seed=4, rate=0.1)
    first_rows = list(first.rows(_rows()))
    second = AnomalyInjector(seed=4, rate=0.1)
    second_rows = list(second.rows(_rows()))
    assert first_rows == second_rows
    assert first.labels == second.labels


def test_zero_rate_injects_nothing() -> None:
    injector = AnomalyInjector(seed=5, rate=0.0)
    original = list(_rows())
    result = list(injector.rows(_rows()))
    assert result == original
    assert injector.labels == []


def test_anomalous_rows_still_respect_the_funnel() -> None:
    injector = AnomalyInjector(seed=6, rate=0.3)
    for row in injector.rows(_rows()):
        assert row.impressions >= row.clicks >= row.conversions >= 0
        assert row.spend >= 0.0


def test_out_of_order_stream_still_has_no_duplicate_labels() -> None:
    """Regression for the ordering hole in `_window_for`: `rows()` advertises
    `Iterable[EventRow]` with no stated ordering precondition, so a window
    opened by a later-processed, earlier-dated row must not be allowed to
    overlap a window an earlier-processed row already opened."""
    catalog = build_catalog(
        seed=5,
        n_advertisers=1,
        campaigns_per_advertiser=1,
        creatives_per_campaign=2,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )
    ordered = list(generate_events(catalog, seed=1))
    shuffled = ordered.copy()
    random.Random(42).shuffle(shuffled)

    injector = AnomalyInjector(seed=3, rate=0.20)
    list(injector.rows(shuffled))

    counts = Counter((label.campaign_id, label.event_date) for label in injector.labels)
    duplicates = {key: count for key, count in counts.items() if count > 1}
    assert not duplicates, f"duplicate campaign-day labels from out-of-order input: {duplicates}"


def test_tracking_error_suppresses_conversions() -> None:
    injector = AnomalyInjector(seed=7, rate=1.0, forced_cause="tracking_error")
    rows = list(injector.rows(_rows()))
    labelled_days = {(label.campaign_id, label.event_date) for label in injector.labels}
    affected = [r for r in rows if (r.campaign_id, r.event_date) in labelled_days]
    assert affected
    assert all(row.conversions == 0 for row in affected)
