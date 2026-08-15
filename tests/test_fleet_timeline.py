"""The fleet timeline is the model's foundation -- if capacity is wrong, everything
downstream is confidently wrong. These tests pin the behaviour that matters."""

from __future__ import annotations

import copy
from datetime import date

import pytest

from anos.data.fleet_timeline import daily_aircraft_hours, fleet_on, timeline
from anos.data.loaders import load_fleet_config


@pytest.fixture
def cfg():
    return copy.deepcopy(load_fleet_config())


def test_anchor_date_matches_in_service_counts(cfg):
    """On the anchor date, nominal counts are exactly what is on the books."""
    snap = fleet_on(cfg["as_of"], config=cfg)
    for code, count in cfg["in_service"].items():
        assert snap.nominal[code] == pytest.approx(count)


def test_available_is_always_below_nominal(cfg):
    """Maintenance and retrofits can only remove capacity, never add it."""
    snap = fleet_on(date(2027, 6, 1), config=cfg)
    for code in snap.nominal:
        assert snap.available[code] <= snap.nominal[code] + 1e-9


def test_deliveries_accumulate_over_time(cfg):
    """A type with an open order grows monotonically across its delivery window."""
    counts = [
        fleet_on(d, config=cfg).nominal["A320neo"]
        for d in (date(2026, 7, 1), date(2027, 7, 1), date(2028, 7, 1), date(2029, 1, 1))
    ]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_delivery_window_completes_exactly(cfg):
    """By the end of its window an order is fully delivered, and no more.

    Uses A321XLR specifically because it has exactly one order tranche in the real
    config -- A350-900 and B737MAX8 each now have two (a 2026-08 top-up on top of
    the original order), so "well past the window" for either of those legitimately
    keeps growing once the second tranche's own window opens.
    """
    order = next(o for o in cfg["orders"] if o["type"] == "A321XLR")
    delivered = fleet_on(order["end"], config=cfg).nominal["A321XLR"]
    expected = cfg["in_service"].get("A321XLR", 0) + order["count"]
    assert delivered == pytest.approx(expected)

    # Well past the window it must not keep growing.
    later = fleet_on(date(2033, 1, 1), config=cfg).nominal["A321XLR"]
    assert later == pytest.approx(expected)


def test_retirements_reduce_the_fleet(cfg):
    """A fully retired type is gone by the end of its retirement window."""
    entry = next(r for r in cfg["retirements"] if r["type"] == "A319")
    assert fleet_on(entry["end"], config=cfg).nominal["A319"] == pytest.approx(0.0)


def test_retrofit_removes_aircraft_only_during_its_window(cfg):
    """Retrofit downtime is live capacity loss, and it ends when the programme does."""
    entry = next(r for r in cfg["retrofits"] if r["type"] == "B787-8")
    during = fleet_on(date(2026, 9, 1), config=cfg)
    after = fleet_on(entry["end"].replace(year=entry["end"].year + 1), config=cfg)

    assert during.in_retrofit.get("B787-8", 0.0) > 0
    assert after.in_retrofit.get("B787-8", 0.0) == 0.0
    assert during.available["B787-8"] < after.available["B787-8"]


def test_probabilistic_mode_never_exceeds_nominal(cfg):
    """Risk-adjusting delivery dates can only delay capacity, never pull it forward."""
    target = date(2028, 6, 1)
    nominal = fleet_on(target, probabilistic=False, config=cfg)
    adjusted = fleet_on(target, probabilistic=True, config=cfg)
    assert adjusted.total_nominal() <= nominal.total_nominal() + 1e-9


def test_low_confidence_programme_is_discounted_hardest(cfg):
    """The 777X, at 0.45 confidence, should be discounted more than the A320neo at 0.90."""
    target = date(2029, 1, 1)
    nominal = fleet_on(target, probabilistic=False, config=cfg)
    adjusted = fleet_on(target, probabilistic=True, config=cfg)

    def shortfall(code: str) -> float:
        return 1 - adjusted.nominal[code] / nominal.nominal[code]

    assert shortfall("B777-9") > shortfall("A320neo")


def test_daily_hours_scale_with_type_utilisation(cfg):
    """Flyable hours are airframes times that type's planned daily utilisation."""
    from anos.data.loaders import load_aircraft_types

    snap = fleet_on(date(2027, 1, 1), config=cfg)
    hours = daily_aircraft_hours(snap)
    types = load_aircraft_types()
    for code, h in hours.items():
        assert h == pytest.approx(snap.available[code] * types[code].max_daily_util_h)


def test_timeline_is_ordered_and_covers_the_range():
    snaps = timeline(date(2026, 7, 1), date(2028, 7, 1), step_days=180)
    assert len(snaps) >= 4
    assert [s.as_of for s in snaps] == sorted(s.as_of for s in snaps)


def test_b737max10_type_loads_and_phases_in():
    """The 2026-08 Wings India top-up added a new type to the fleet -- confirm it
    actually loads and its order eventually delivers."""
    from anos.data.loaders import load_aircraft_types

    types = load_aircraft_types()
    assert "B737MAX10" in types
    assert types["B737MAX10"].generation in ("neo", "new_widebody", "legacy", "legacy_widebody")

    early = fleet_on(date(2028, 6, 1)).nominal.get("B737MAX10", 0.0)
    later = fleet_on(date(2029, 6, 1)).nominal.get("B737MAX10", 0.0)
    assert early == pytest.approx(0.0)
    assert later > 0.0


def test_every_type_resolves_an_unscheduled_downtime_rate():
    """Params.unscheduled_downtime_rate is keyed by generation now, not a single
    network-wide number -- every real aircraft type's generation must resolve."""
    from anos.config import load_params
    from anos.data.loaders import load_aircraft_types

    par = load_params()
    types = load_aircraft_types()
    for code, ac in types.items():
        rate = par.unscheduled_downtime_rate(ac.generation)
        assert 0.0 <= rate <= 1.0, f"{code} ({ac.generation}) resolved an implausible rate {rate}"


def test_legacy_generations_carry_more_downtime_risk_than_neo():
    """Grounded in Air India's own documented spares-funding history -- see
    cost_params.yaml's operations.unscheduled_downtime_rate comment."""
    from anos.config import load_params

    par = load_params()
    assert par.unscheduled_downtime_rate("legacy") > par.unscheduled_downtime_rate("neo")
    assert par.unscheduled_downtime_rate("legacy_widebody") > par.unscheduled_downtime_rate(
        "new_widebody"
    )
