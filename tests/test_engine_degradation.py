"""Tests for anos.costs.engine_degradation: the age-driven fuel-burn-degradation
curve, its fleet-age estimation, and its opt-in wiring into the cost engine."""

from __future__ import annotations

from datetime import date

import pytest

from anos.costs.economics import build_economics, leg_economics
from anos.costs.engine_degradation import (
    DegradationAssumptions,
    degradation_pct,
    degraded_fuel_burn_kgh,
    estimate_fleet_average_age,
    load_degradation_assumptions,
    sensitivity_bands,
)
from anos.data.loaders import load_aircraft_types, load_fleet_config, load_markets
from anos.optimize.fleet_assignment import solve_network

CURVE = DegradationAssumptions(raw={"curve": {"d_max": 0.06, "age_ref_years": 20}})


# -- degradation_pct: pure curve -------------------------------------------------


def test_degradation_is_zero_at_age_zero():
    assert degradation_pct(0, CURVE) == 0.0


def test_degradation_is_zero_for_negative_age():
    assert degradation_pct(-1, CURVE) == 0.0


def test_degradation_matches_the_calibration_point():
    assert degradation_pct(20, CURVE) == pytest.approx(0.06)


def test_degradation_increases_monotonically_with_age():
    ages = [0, 1, 5, 10, 20, 40]
    values = [degradation_pct(a, CURVE) for a in ages]
    assert values == sorted(values)
    assert values[-1] > values[0]


def test_real_assumptions_load_and_resolve_sensibly():
    a = load_degradation_assumptions()
    assert 0.0 < a.d_max < 0.20
    assert a.age_ref_years > 0


# -- estimate_fleet_average_age --------------------------------------------------


def _fleet_config(**overrides) -> dict:
    base = {
        "as_of": date(2026, 1, 1),
        "in_service": {},
        "orders": [],
        "retirements": [],
        "retrofits": [],
    }
    base.update(overrides)
    return base


def test_baseline_only_type_ages_linearly_with_elapsed_time():
    assumptions = DegradationAssumptions(
        raw={"curve": {"d_max": 0.06, "age_ref_years": 20}, "baseline_avg_age_years": {"T1": 5.0}}
    )
    cfg = _fleet_config(in_service={"T1": 10})

    age_at_anchor = estimate_fleet_average_age("T1", date(2026, 1, 1), fleet_config=cfg, assumptions=assumptions)
    age_3y_later = estimate_fleet_average_age("T1", date(2029, 1, 1), fleet_config=cfg, assumptions=assumptions)

    assert age_at_anchor == pytest.approx(5.0, abs=0.01)
    assert age_3y_later == pytest.approx(8.0, abs=0.02)


def test_type_with_no_aircraft_at_all_has_zero_age():
    assumptions = DegradationAssumptions(raw={"curve": {"d_max": 0.06, "age_ref_years": 20}, "baseline_avg_age_years": {}})
    cfg = _fleet_config(in_service={})
    assert estimate_fleet_average_age("GHOST", date(2026, 1, 1), fleet_config=cfg, assumptions=assumptions) == 0.0


def test_growing_fleet_pulls_the_average_age_down():
    """A type mid-delivery-ramp has a younger average age than a baseline-only type
    of the same starting age, because the new deliveries are diluting it."""
    assumptions = DegradationAssumptions(
        raw={"curve": {"d_max": 0.06, "age_ref_years": 20}, "baseline_avg_age_years": {"T1": 5.0}}
    )
    baseline_only_cfg = _fleet_config(in_service={"T1": 10})
    growing_cfg = _fleet_config(
        in_service={"T1": 10},
        orders=[{"type": "T1", "count": 10, "start": date(2026, 1, 1), "end": date(2027, 12, 31), "confidence": 1.0}],
    )

    target = date(2029, 1, 1)
    baseline_only_age = estimate_fleet_average_age("T1", target, fleet_config=baseline_only_cfg, assumptions=assumptions)
    growing_age = estimate_fleet_average_age("T1", target, fleet_config=growing_cfg, assumptions=assumptions)

    assert growing_age < baseline_only_age


def test_retired_type_has_zero_age_once_fully_retired():
    assumptions = DegradationAssumptions(
        raw={"curve": {"d_max": 0.06, "age_ref_years": 20}, "baseline_avg_age_years": {"T1": 15.0}}
    )
    cfg = _fleet_config(
        in_service={"T1": 6},
        retirements=[{"type": "T1", "count": 6, "start": date(2026, 1, 1), "end": date(2027, 12, 31)}],
    )
    age_after_retirement = estimate_fleet_average_age("T1", date(2029, 1, 1), fleet_config=cfg, assumptions=assumptions)
    assert age_after_retirement == 0.0


def test_real_fleet_ages_resolve_for_every_type():
    """Every real aircraft type resolves *some* age (possibly 0.0 if fully retired
    or not yet delivered) without raising, for a spread of real horizon dates."""
    types = load_aircraft_types()
    cfg = load_fleet_config()
    for code in types:
        for target in (date(2027, 3, 1), date(2030, 3, 1), date(2032, 3, 1)):
            age = estimate_fleet_average_age(code, target, fleet_config=cfg)
            assert age >= 0.0


# -- opt-in wiring: non-regression -----------------------------------------------


def test_omitted_degraded_fuel_burn_preserves_nominal_fuel_burn():
    """The core non-regression guarantee: leg_economics with no degradation info
    at all must use the type's plain, undiscounted fuel_burn_kgh."""
    market = next(m for m in load_markets() if m.market_id == "DEL-BOM")
    ac = load_aircraft_types()["A320neo"]

    without_kw = leg_economics(market, ac)
    with_none = leg_economics(market, ac, degraded_fuel_burn=None)
    with_empty = leg_economics(market, ac, degraded_fuel_burn={})
    with_other_type_only = leg_economics(market, ac, degraded_fuel_burn={"SOME-OTHER-TYPE": 9999.0})

    assert without_kw.cost_per_round_trip_usd == with_none.cost_per_round_trip_usd
    assert without_kw.cost_per_round_trip_usd == with_empty.cost_per_round_trip_usd
    assert without_kw.cost_per_round_trip_usd == with_other_type_only.cost_per_round_trip_usd


def test_build_economics_default_matches_explicit_none():
    markets = [m for m in load_markets() if m.market_id in ("DEL-BOM", "BLR-MAA")]
    default_econ = build_economics(markets)
    explicit_none_econ = build_economics(markets, degraded_fuel_burn=None)
    assert {k: v.cost_per_round_trip_usd for k, v in default_econ.items()} == {
        k: v.cost_per_round_trip_usd for k, v in explicit_none_econ.items()
    }


# -- opt-in wiring: real effect ---------------------------------------------------


def test_degraded_fuel_burn_is_never_below_nominal():
    types = load_aircraft_types()
    for ac in types.values():
        for age in (0.0, 5.0, 20.0, 40.0):
            assert degraded_fuel_burn_kgh(ac, age) >= ac.fuel_burn_kgh


def test_enabling_engine_degradation_increases_cost_for_an_aged_fleet():
    """A target date well past the fleet's anchor should show a real, material cost
    increase once degradation is enabled -- not just numerical noise."""
    target = date(2031, 3, 1)
    baseline = solve_network(target)
    degraded = solve_network(target, enable_engine_degradation=True)

    assert degraded.total_cost_usd > baseline.total_cost_usd
    assert degraded.total_contribution_usd < baseline.total_contribution_usd
    # The effect should be a real, few-percent-scale impact, not a rounding artifact.
    relative_increase = (degraded.total_cost_usd - baseline.total_cost_usd) / baseline.total_cost_usd
    assert relative_increase > 0.001


# -- sensitivity bands -------------------------------------------------------------


def test_sensitivity_bands_span_low_to_high_in_order():
    bands = sensitivity_bands(load_degradation_assumptions())
    assert bands["low"].d_max < bands["central"].d_max < bands["high"].d_max


def test_sensitivity_bands_scale_the_curve_not_just_the_label():
    bands = sensitivity_bands(load_degradation_assumptions())
    age = 20.0
    low_pct = degradation_pct(age, bands["low"])
    central_pct = degradation_pct(age, bands["central"])
    high_pct = degradation_pct(age, bands["high"])
    assert low_pct < central_pct < high_pct


def test_scaled_assumptions_leave_other_fields_untouched():
    a = load_degradation_assumptions()
    scaled = a.scaled(2.0)
    assert scaled.age_ref_years == a.age_ref_years
    assert scaled.d_max == pytest.approx(a.d_max * 2.0)


def test_higher_sensitivity_band_costs_more_at_an_aged_target():
    """Real re-solves under low vs. high assumptions should show a monotonic
    ordering of the fuel-cost impact, not just the curve parameters in isolation."""
    target = date(2031, 3, 1)
    markets = [m for m in load_markets() if m.market_id in ("DEL-BOM", "BLR-MAA", "DEL-BLR")]
    bands = sensitivity_bands(load_degradation_assumptions())

    costs = {}
    for name, band in bands.items():
        plan = solve_network(
            target, markets=markets, enable_engine_degradation=True, degradation_assumptions=band
        )
        costs[name] = plan.total_cost_usd

    assert costs["low"] <= costs["central"] <= costs["high"]
