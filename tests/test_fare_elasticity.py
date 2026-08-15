"""Fare elasticity: the opt-in demand response layered on top of the frequency
ceiling in `anos.forecast.demand`."""

from __future__ import annotations

import pytest

from anos.config import load_params
from anos.forecast.fare_elasticity import (
    demand_multiplier_for_fare,
    fare_levels,
    market_elasticity,
)
from tests.test_demand import make_market


def test_reference_fare_multiplier_is_neutral():
    """At the reference fare, elasticity must not move demand at all -- this is the
    anchor that keeps `enable_fare_elasticity=False` byte-identical to today."""
    market = make_market(avg_fare_usd=100.0)
    assert demand_multiplier_for_fare(market, 100.0) == pytest.approx(1.0)


def test_lower_fare_increases_demand_multiplier():
    market = make_market(avg_fare_usd=100.0, business_mix=0.3)
    assert demand_multiplier_for_fare(market, 85.0) > 1.0


def test_higher_fare_decreases_demand_multiplier():
    market = make_market(avg_fare_usd=100.0, business_mix=0.3)
    assert demand_multiplier_for_fare(market, 115.0) < 1.0


def test_business_heavy_markets_are_less_fare_sensitive():
    par = load_params()
    business = make_market(business_mix=0.95)
    leisure = make_market(business_mix=0.05)
    assert abs(market_elasticity(business, par)) < abs(market_elasticity(leisure, par))


def test_business_heavy_markets_swing_less_on_a_fare_cut():
    business = make_market(avg_fare_usd=100.0, business_mix=0.95)
    leisure = make_market(avg_fare_usd=100.0, business_mix=0.05)
    business_lift = demand_multiplier_for_fare(business, 85.0) - 1.0
    leisure_lift = demand_multiplier_for_fare(leisure, 85.0) - 1.0
    assert leisure_lift > business_lift > 0


def test_multiplier_is_clipped_to_the_configured_band():
    par = load_params()
    lo, hi = par.fare_elasticity_multiplier_band
    market = make_market(avg_fare_usd=100.0, business_mix=0.0)
    assert demand_multiplier_for_fare(market, 1.0, par) <= hi
    assert demand_multiplier_for_fare(market, 100000.0, par) >= lo


def test_fare_levels_are_relative_to_the_markets_own_reference_fare():
    par = load_params()
    market = make_market(avg_fare_usd=200.0)
    levels = fare_levels(market, par)
    assert levels == [round(200.0 * m, 2) for m in par.fare_multipliers]
    assert 200.0 in levels  # the reference multiplier (1.00) must be one of the options
