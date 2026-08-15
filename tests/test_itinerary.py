"""Connection and rotation candidate builders.

`build_connection_candidates` and `build_rotation_candidates` are pure functions over
markets.csv -- no solver involved -- so they're tested directly against the real
reference data, which is also what lets these tests assert on concrete market ids
rather than synthetic fixtures.
"""

from __future__ import annotations

from datetime import date

import pytest

from anos.data.loaders import load_markets
from anos.models import Market, MarketPlan, NetworkPlan
from anos.network.itinerary import (
    build_connection_candidates,
    build_rotation_candidates,
    low_load_factor_squad,
)


def test_build_connection_candidates_finds_maa_lhr_via_del():
    """MAA-LHR has no nonstop, but DEL-MAA and DEL-LHR both exist -- the concrete
    gap this whole feature exists to close."""
    candidates = build_connection_candidates(load_markets(), hubs=["DEL"])
    by_market = {c.market_id: c for c in candidates}
    assert "MAA-LHR" in by_market
    c = by_market["MAA-LHR"]
    assert c.hub == "DEL"
    assert c.feeder_market_id == "DEL-MAA"
    assert c.trunk_market_id == "DEL-LHR"


def test_build_connection_candidates_excludes_direct_hub_markets():
    """DEL-LHR itself must never be treated as a connection through DEL: neither
    endpoint of a connecting market may be the hub it's supposedly connecting via."""
    candidates = build_connection_candidates(load_markets(), hubs=["DEL"])
    markets_by_id = {m.market_id: m for m in load_markets()}
    for c in candidates:
        market = markets_by_id[c.market_id]
        assert "DEL" not in (market.origin, market.destination)
    assert not any(c.market_id == "DEL-LHR" for c in candidates)


def test_build_connection_candidates_requires_both_legs_present():
    markets = [
        Market(
            market_id="DEL-XXX",
            origin="DEL",
            destination="XXX",
            region="domestic_trunk",
            base_daily_demand=100.0,
            avg_fare_usd=100.0,
            business_mix=0.5,
            competition_index=0.5,
            min_daily_freq=0,
            max_daily_freq=5,
            seasonality_profile="domestic_business",
            strategic=False,
        ),
        Market(
            market_id="XXX-LHR",
            origin="XXX",
            destination="LHR",
            region="longhaul_europe",
            base_daily_demand=50.0,
            avg_fare_usd=500.0,
            business_mix=0.5,
            competition_index=0.5,
            min_daily_freq=0,
            max_daily_freq=1,
            seasonality_profile="intl_summer",
            strategic=False,
        ),
    ]
    # No "DEL-LHR" trunk market exists, so no connection candidate should be built.
    assert build_connection_candidates(markets, hubs=["DEL"]) == []


def test_star_hub_is_the_same_builder_with_a_different_hub():
    """The 'star model' claim: a secondary hub is a parameter, not new code."""
    markets = load_markets()
    del_candidates = build_connection_candidates(markets, hubs=["DEL"])
    blr_candidates = build_connection_candidates(markets, hubs=["BLR"])
    combined = build_connection_candidates(markets, hubs=["DEL", "BLR"])
    assert {c.connection_id for c in combined} == {c.connection_id for c in del_candidates} | {
        c.connection_id for c in blr_candidates
    }


def test_build_rotation_candidates_is_bounded_not_a_general_search():
    """k squad markets sharing a hub must give exactly k*(k-1) ordered-pair loops --
    never a search that could grow unpredictably as the squad grows."""
    markets = load_markets()
    squad = ["DEL-NAG", "DEL-IDR", "DEL-IXM"]  # 3 squad markets, all DEL-origin
    candidates = build_rotation_candidates(markets, squad, hubs=["DEL"], max_stops=2)
    assert len(candidates) == 3 * 2  # k*(k-1) with k=3


def test_build_rotation_candidates_ignores_markets_outside_the_squad():
    markets = load_markets()
    candidates = build_rotation_candidates(markets, ["DEL-NAG"], hubs=["DEL"], max_stops=2)
    assert candidates == []  # a single spoke cannot form a loop


def test_build_rotation_candidates_only_pairs_spokes_sharing_a_hub():
    markets = load_markets()
    # DEL-NAG is DEL-origin; nothing here is BOM-origin, so hub=BOM must yield nothing.
    candidates = build_rotation_candidates(markets, ["DEL-NAG", "DEL-IDR"], hubs=["BOM"])
    assert candidates == []


def test_build_rotation_candidates_rejects_more_than_two_stops():
    with pytest.raises(NotImplementedError):
        build_rotation_candidates(load_markets(), ["DEL-NAG"], hubs=["DEL"], max_stops=3)


def test_low_load_factor_squad_only_includes_served_thin_markets():
    plan = NetworkPlan(
        as_of=date(2026, 8, 3),
        month=8,
        market_plans=[
            MarketPlan("A-B", {"X": 1}, 1, 100, 40, 40, 1000.0, 900.0),  # LF 40%
            MarketPlan("C-D", {"X": 1}, 1, 100, 90, 90, 1000.0, 900.0),  # LF 90%
            MarketPlan("E-F", {}, 0, 0, 0, 0, 0.0, 0.0),  # unserved
        ],
        fleet_used={},
        fleet_available={},
        solver_status="OPTIMAL",
        objective_usd=0.0,
        solve_seconds=0.0,
    )
    assert low_load_factor_squad(plan, threshold=0.50) == ["A-B"]
