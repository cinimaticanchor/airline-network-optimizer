"""Pure-function tests for the multi-year demand growth curve."""

from __future__ import annotations

import pytest

from anos.data.loaders import load_markets
from anos.forecast.demand_growth import (
    GrowthCurve,
    annual_growth_rate,
    cumulative_growth_factor,
    grow_market,
    grow_markets,
    load_growth_assumptions,
)

CURVE = GrowthCurve(near_term_cagr=0.10, long_run_cagr=0.02, decay_half_life_years=1.0)


def test_rate_at_zero_elapsed_years_is_the_near_term_rate():
    assert annual_growth_rate(CURVE, 0) == pytest.approx(0.10)


def test_rate_decays_toward_the_long_run_rate():
    """One half-life in, the gap to the long-run rate has halved."""
    rate_at_one_half_life = annual_growth_rate(CURVE, 1)
    assert rate_at_one_half_life == pytest.approx(0.02 + (0.10 - 0.02) * 0.5)


def test_rate_converges_to_long_run_far_out():
    far_rate = annual_growth_rate(CURVE, 50)
    assert far_rate == pytest.approx(0.02, abs=1e-6)


def test_cumulative_growth_factor_is_one_with_no_elapsed_years():
    assert cumulative_growth_factor(CURVE, 2026, 2026) == pytest.approx(1.0)


def test_cumulative_growth_factor_matches_hand_compounded_value():
    # Year 0 (2026->2027) grows at the near-term rate; year 1 (2027->2028) grows at
    # the once-decayed rate -- hand-computed here independently of the implementation.
    expected = (1 + 0.10) * (1 + (0.02 + (0.10 - 0.02) * 0.5))
    assert cumulative_growth_factor(CURVE, 2026, 2028) == pytest.approx(expected)


def test_cumulative_growth_factor_rejects_going_backwards():
    with pytest.raises(ValueError):
        cumulative_growth_factor(CURVE, 2030, 2026)


def test_every_real_market_region_resolves_a_bucket_and_curve():
    assumptions = load_growth_assumptions()
    for market in load_markets():
        bucket = assumptions.bucket_for_region(market.region)
        curve = assumptions.curve_for_region(market.region)
        assert bucket
        assert curve.near_term_cagr > 0
        assert curve.decay_half_life_years > 0


def test_grow_market_only_scales_base_daily_demand():
    market = load_markets()[0]
    assumptions = load_growth_assumptions()
    grown = grow_market(market, assumptions.anchor_year + 3, assumptions=assumptions)

    assert grown.base_daily_demand > market.base_daily_demand
    assert grown.market_id == market.market_id
    assert grown.avg_fare_usd == market.avg_fare_usd
    assert grown.region == market.region


def test_grow_market_at_anchor_year_is_unchanged():
    market = load_markets()[0]
    assumptions = load_growth_assumptions()
    grown = grow_market(market, assumptions.anchor_year, assumptions=assumptions)
    assert grown.base_daily_demand == pytest.approx(market.base_daily_demand)


def test_grow_markets_grows_every_market_in_the_list():
    markets = load_markets()
    assumptions = load_growth_assumptions()
    grown = grow_markets(markets, assumptions.anchor_year + 5, assumptions=assumptions)

    assert len(grown) == len(markets)
    for original, g in zip(markets, grown, strict=True):
        assert g.base_daily_demand >= original.base_daily_demand
