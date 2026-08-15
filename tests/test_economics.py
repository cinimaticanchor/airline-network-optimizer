"""Route unit economics and aircraft eligibility."""

from __future__ import annotations

import pytest

from anos.costs.economics import build_economics, leg_economics, market_distance_km
from anos.data.loaders import load_aircraft_types, load_markets
from anos.models import great_circle_km
from tests.test_demand import make_market


def test_great_circle_matches_known_distance():
    """Delhi to Mumbai is about 1150 km."""
    assert great_circle_km(28.5562, 77.1000, 19.0887, 72.8679) == pytest.approx(1150, abs=40)


def test_great_circle_is_zero_for_identical_points():
    assert great_circle_km(28.5562, 77.1, 28.5562, 77.1) == pytest.approx(0.0)


def test_short_haul_aircraft_cannot_fly_long_haul():
    """An A320neo has no business on Delhi-San Francisco."""
    types = load_aircraft_types()
    market = make_market(market_id="DEL-SFO", origin="DEL", destination="SFO")
    econ = leg_economics(market, types["A320neo"])
    assert not econ.eligible
    assert "range" in econ.ineligible_reason


def test_widebody_blocked_from_incapable_airport():
    """Pune cannot take a widebody, so no widebody may be assigned there."""
    types = load_aircraft_types()
    market = make_market(market_id="DEL-PNQ", origin="DEL", destination="PNQ")
    econ = leg_economics(market, types["B787-9"])
    assert not econ.eligible
    assert "PNQ" in econ.ineligible_reason


def test_long_range_type_is_eligible_for_long_haul():
    types = load_aircraft_types()
    market = make_market(market_id="DEL-JFK", origin="DEL", destination="JFK")
    assert leg_economics(market, types["B787-9"]).eligible


def test_cost_and_time_rise_with_distance():
    types = load_aircraft_types()
    short = make_market(market_id="DEL-JAI", origin="DEL", destination="JAI")
    long = make_market(market_id="DEL-BLR", origin="DEL", destination="BLR")

    a, b = (leg_economics(m, types["A320neo"]) for m in (short, long))
    assert b.distance_km > a.distance_km
    assert b.cost_per_round_trip_usd > a.cost_per_round_trip_usd
    assert b.block_hours_round_trip > a.block_hours_round_trip


def test_aircraft_hours_include_ground_time():
    """A round trip ties up the aircraft for longer than it is airborne."""
    types = load_aircraft_types()
    market = make_market(market_id="DEL-BOM", origin="DEL", destination="BOM")
    econ = leg_economics(market, types["A320neo"])
    assert econ.aircraft_hours_consumed > econ.block_hours_round_trip


def test_larger_aircraft_costs_more_per_trip_but_less_per_seat():
    """The core fleet-assignment trade-off: upgauging buys unit cost."""
    types = load_aircraft_types()
    market = make_market(market_id="DEL-BOM", origin="DEL", destination="BOM")

    small = leg_economics(market, types["A320neo"])
    large = leg_economics(market, types["A321neo"])

    assert large.cost_per_round_trip_usd > small.cost_per_round_trip_usd
    small_casm = small.cost_per_round_trip_usd / small.seats_round_trip
    large_casm = large.cost_per_round_trip_usd / large.seats_round_trip
    assert large_casm < small_casm


def test_neo_burns_less_than_ceo_on_the_same_sector():
    types = load_aircraft_types()
    market = make_market(market_id="DEL-BOM", origin="DEL", destination="BOM")
    neo = leg_economics(market, types["A320neo"])
    ceo = leg_economics(market, types["A320ceo"])
    assert neo.cost_per_round_trip_usd / neo.seats_round_trip < (
        ceo.cost_per_round_trip_usd / ceo.seats_round_trip
    )


def test_build_economics_returns_only_eligible_pairings():
    econ = build_economics()
    assert econ
    assert all(e.eligible for e in econ.values())


def test_every_market_has_at_least_one_eligible_type():
    """A market no aircraft can serve is a data error, not an optimisation outcome."""
    econ = build_economics()
    served = {mid for (mid, _) in econ}
    missing = [m.market_id for m in load_markets() if m.market_id not in served]
    assert not missing, f"markets with no eligible aircraft: {missing}"


def test_market_distance_helper_agrees_with_leg_economics():
    types = load_aircraft_types()
    market = load_markets()[0]
    econ = leg_economics(market, types["A320neo"])
    assert market_distance_km(market) == pytest.approx(econ.distance_km)
