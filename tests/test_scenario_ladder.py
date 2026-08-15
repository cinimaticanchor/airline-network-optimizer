"""Small-scale integration test for anos.planning.scenario_ladder, against real
reference data restricted to a couple of markets to keep the solve fast."""

from __future__ import annotations

from datetime import date

import pytest

from anos.data.fleet_timeline import fleet_on
from anos.data.loaders import load_markets
from anos.forecast.demand_growth import GrowthAssumptions
from anos.planning.scenario_ladder import build_checkpoints, run_ladder

ANCHOR_YEAR = 2027


def _small_market_set() -> list:
    """A couple of real markets, trimmed so the solve stays quick."""
    return [m for m in load_markets() if m.market_id in ("DEL-BOM", "BLR-MAA")]


def _zero_growth_assumptions(markets: list) -> GrowthAssumptions:
    """A GrowthAssumptions where every region in `markets` maps to a flat 0% curve,
    isolating "does the ladder mechanically work" from the real growth numbers."""
    regions = {m.region for m in markets}
    return GrowthAssumptions(
        raw={
            "anchor_year": ANCHOR_YEAR,
            "buckets": {
                "flat": {
                    "near_term_cagr": 0.0,
                    "long_run_cagr": 0.0,
                    "decay_half_life_years": 5.0,
                }
            },
            "region_bucket_map": {region: "flat" for region in regions},
        }
    )


def test_build_checkpoints_covers_the_range_inclusive():
    checkpoints = build_checkpoints(2027, 2029, month=3)
    assert checkpoints == [date(2027, 3, 1), date(2028, 3, 1), date(2029, 3, 1)]


def test_build_checkpoints_rejects_a_backwards_range():
    with pytest.raises(ValueError):
        build_checkpoints(2029, 2027)


def test_zero_growth_holds_contribution_flat():
    markets = _small_market_set()
    assert len(markets) == 2
    growth = _zero_growth_assumptions(markets)

    result = run_ladder(2027, 2029, markets=markets, growth=growth)
    series = result.contribution_series()

    assert len(series) == 3
    # CP-SAT's own relative gap moves results by roughly 0.5-1% run to run; a flat
    # demand curve should hold well inside that band across the whole horizon.
    baseline = series[0]
    for value in series:
        assert value == pytest.approx(baseline, rel=0.05)


def test_checkpoint_fleet_matches_a_direct_fleet_on_call():
    markets = _small_market_set()
    growth = _zero_growth_assumptions(markets)
    result = run_ladder(2027, 2028, markets=markets, growth=growth)

    for checkpoint in result.checkpoints:
        expected = fleet_on(checkpoint.as_of)
        assert checkpoint.fleet.nominal == expected.nominal
        assert checkpoint.fleet.available == expected.available


def test_grown_markets_are_recorded_on_each_checkpoint():
    markets = _small_market_set()
    growth = _zero_growth_assumptions(markets)
    result = run_ladder(2027, 2027, markets=markets, growth=growth)

    checkpoint = result.checkpoints[0]
    assert {m.market_id for m in checkpoint.grown_markets} == {"DEL-BOM", "BLR-MAA"}
