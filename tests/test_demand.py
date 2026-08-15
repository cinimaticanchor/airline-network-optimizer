"""Demand behaviour: seasonality and the frequency/schedule-quality curve."""

from __future__ import annotations

import pytest

from anos.config import load_params
from anos.forecast.demand import (
    demand_curve,
    effective_demand,
    schedule_quality,
    seasonal_demand,
)
from anos.models import Market


def make_market(**overrides) -> Market:
    base = dict(
        market_id="TEST",
        origin="DEL",
        destination="BOM",
        region="domestic_trunk",
        base_daily_demand=1000.0,
        avg_fare_usd=100.0,
        business_mix=0.5,
        competition_index=0.5,
        min_daily_freq=0,
        max_daily_freq=10,
        seasonality_profile="domestic_business",
        strategic=False,
    )
    base.update(overrides)
    return Market(**base)


def test_reference_frequency_reproduces_base_demand():
    """The whole curve is normalised so the reference schedule returns the quoted
    demand. If this drifts, every demand figure in markets.csv silently changes
    meaning."""
    params = load_params()
    market = make_market()
    assert schedule_quality(market, int(params.frequency_ref), params) == pytest.approx(1.0)


def test_no_service_carries_nobody():
    assert schedule_quality(make_market(), 0) == 0.0
    assert effective_demand(make_market(), 0, month=3) == 0.0


def test_demand_rises_with_frequency_but_with_diminishing_returns():
    """The S-curve: more frequency always helps, each increment helps less."""
    market = make_market()
    values = [schedule_quality(market, f) for f in range(1, 9)]
    assert values == sorted(values)

    # Deliberately uneven: each value is paired with its successor.
    gains = [b - a for a, b in zip(values, values[1:], strict=False)]
    assert gains == sorted(gains, reverse=True)


def test_uplift_is_capped():
    """Frequency alone cannot grow a market without limit."""
    params = load_params()
    market = make_market(business_mix=1.0, competition_index=0.0)
    ceiling = 1 + params.max_frequency_uplift
    assert schedule_quality(market, 200, params) <= ceiling + 1e-9


def test_business_markets_reward_frequency_more_than_leisure():
    business = make_market(business_mix=0.9, competition_index=0.3)
    leisure = make_market(business_mix=0.1, competition_index=0.3)
    assert schedule_quality(business, 8) > schedule_quality(leisure, 8)


def test_competition_damps_the_uplift():
    """In a contested market, added frequency gets matched and captures less."""
    quiet = make_market(competition_index=0.0)
    contested = make_market(competition_index=1.0)
    assert schedule_quality(quiet, 8) > schedule_quality(contested, 8)


def test_seasonality_moves_demand_and_averages_near_one():
    """Profiles redistribute demand across the year rather than inventing it."""
    params = load_params()
    market = make_market(seasonality_profile="leisure_winter")
    monthly = [seasonal_demand(market, m, params) for m in range(1, 13)]

    assert max(monthly) > min(monthly)
    assert sum(monthly) / 12 == pytest.approx(market.base_daily_demand, rel=0.05)


def test_winter_and_summer_profiles_peak_in_opposite_seasons():
    winter = make_market(seasonality_profile="leisure_winter")
    summer = make_market(seasonality_profile="leisure_summer")
    assert seasonal_demand(winter, 1) > seasonal_demand(winter, 6)
    assert seasonal_demand(summer, 6) > seasonal_demand(summer, 1)


def test_curve_covers_every_feasible_frequency():
    market = make_market(max_daily_freq=7)
    curve = demand_curve(market, month=3)
    assert set(curve) == set(range(0, 8))
    assert curve[0] == 0.0
