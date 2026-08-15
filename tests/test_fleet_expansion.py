"""Tests for anos.planning.fleet_expansion: the NPV algorithm that turns a
persistent fleet-utilisation signal into a sized, financially-evaluated candidate
order.

Integration fixtures deliberately shrink a fleet to a single market so the solves
stay fast and the intended effect (genuine shortage vs. genuine slot saturation)
dominates any residual solver noise.
"""

from __future__ import annotations

import copy
from datetime import date

import pytest

from anos.data.loaders import load_fleet_config, load_markets
from anos.models import Market
from anos.planning.constraint_detection import detect_signals
from anos.planning.fleet_expansion import (
    evaluate_candidate,
    load_expansion_assumptions,
    net_present_value,
    payback_year_of,
    recommend_expansion,
    size_candidate,
)
from anos.planning.scenario_ladder import run_ladder

DEL_BOM_MARKET = next(m for m in load_markets() if m.market_id == "DEL-BOM")

# IXM's real daily_slot_cap is 12 -- small enough to bind well before demand does.
SLOT_BOUND_MARKET = Market(
    market_id="IXM-PAT-TEST",
    origin="IXM",
    destination="PAT",
    region="domestic_regional",
    base_daily_demand=3000.0,
    avg_fare_usd=100.0,
    business_mix=0.2,
    competition_index=0.3,
    min_daily_freq=0,
    max_daily_freq=12,
    seasonality_profile="flat",
    strategic=False,
)


def _single_type_config(type_code: str, count: int) -> dict:
    """A fleet config with exactly one type in service and nothing on order --
    everything else in `aircraft_types.yaml` sits at zero availability."""
    cfg = copy.deepcopy(load_fleet_config())
    cfg["in_service"] = {type_code: count}
    cfg["orders"] = []
    cfg["retirements"] = []
    cfg["retrofits"] = []
    return cfg


# -- pure arithmetic: net_present_value -----------------------------------------


def test_npv_of_a_flow_at_the_base_year_is_undiscounted():
    assert net_present_value({2027: 1000.0}, discount_rate=0.09, base_year=2027) == pytest.approx(1000.0)


def test_npv_discounts_future_flows_down():
    value = net_present_value({2028: 1000.0}, discount_rate=0.10, base_year=2027)
    assert value == pytest.approx(1000.0 / 1.10)


def test_npv_sums_multiple_years():
    flows = {2027: 100.0, 2028: 100.0}
    expected = 100.0 + 100.0 / 1.09
    assert net_present_value(flows, discount_rate=0.09, base_year=2027) == pytest.approx(expected)


def test_npv_of_an_all_negative_flow_is_negative():
    assert net_present_value({2027: -500.0, 2028: -500.0}, discount_rate=0.05, base_year=2027) < 0


# -- pure arithmetic: payback_year_of --------------------------------------------


def test_payback_year_is_the_first_year_cumulative_flow_turns_non_negative():
    flows = {2027: -100.0, 2028: -50.0, 2029: 80.0, 2030: 90.0}
    # cumulative: -100, -150, -70, +20 -- first non-negative at 2030.
    assert payback_year_of(flows) == 2030


def test_payback_year_is_none_if_never_recovered():
    assert payback_year_of({2027: -100.0, 2028: -10.0}) is None


def test_payback_year_handles_immediate_recovery():
    assert payback_year_of({2027: 50.0}) == 2027


def test_payback_year_processes_years_chronologically_not_by_insertion_order():
    # Chronological cumulative is -50, -110, -10 -- never recovers. An
    # implementation iterating insertion order would hit 2029's +100 first and
    # wrongly report a payback.
    flows = {2029: 100.0, 2027: -50.0, 2028: -60.0}
    assert payback_year_of(flows) is None


# -- ExpansionAssumptions ---------------------------------------------------------


def test_expansion_assumptions_load_sensible_real_values():
    a = load_expansion_assumptions()
    assert 0.0 < a.discount_rate_annual < 0.30
    assert a.fleet_utilisation_threshold > a.target_utilisation_ceiling
    assert a.max_candidate_count >= a.min_candidate_count >= 1


def test_lead_time_uses_named_overrides_before_the_body_default():
    a = load_expansion_assumptions()
    assert a.lead_time_years("B777-9", "wide") == pytest.approx(5.0)
    assert a.lead_time_years("A350-1000", "wide") == pytest.approx(4.5)
    assert a.lead_time_years("B737MAX10", "narrow") == pytest.approx(3.5)
    # A type with no override just falls back to its body's default.
    assert a.lead_time_years("A320neo", "narrow") == pytest.approx(2.5)
    assert a.lead_time_years("B787-8", "wide") == pytest.approx(4.0)


# -- integration: a genuinely undersized fleet must find a profitable order -----


@pytest.fixture(scope="module")
def undersized_fleet_plan():
    """DEL-BOM (2,600 daily demand, real market) served by just 2 A320neo -- a
    severe, unambiguous shortage with no slot bottleneck in the way."""
    cfg = _single_type_config("A320neo", 2)
    return recommend_expansion(2027, 2029, markets=[DEL_BOM_MARKET], fleet_config=cfg)


def test_undersized_fleet_produces_at_least_one_accepted_order(undersized_fleet_plan):
    assert undersized_fleet_plan.accepted_orders
    assert all(o.type_code == "A320neo" for o in undersized_fleet_plan.accepted_orders)


def test_undersized_fleet_accepted_orders_are_positive_npv(undersized_fleet_plan):
    accepted = {o for o in undersized_fleet_plan.accepted_orders}
    accepted_evaluations = [e for e in undersized_fleet_plan.evaluations if e.candidate in accepted]
    assert accepted_evaluations
    assert all(e.npv_usd > 0 and e.recommended for e in accepted_evaluations)


# -- integration: a slot-bound market must NOT get an order ---------------------


@pytest.fixture(scope="module")
def slot_bound_plan():
    """8 A320neo already fly this market right up to IXM's real 12/day slot cap --
    genuinely full, but no amount of extra fleet can add another departure."""
    cfg = _single_type_config("A320neo", 8)
    return recommend_expansion(2027, 2029, markets=[SLOT_BOUND_MARKET], fleet_config=cfg)


def test_slot_bound_market_gets_no_accepted_order(slot_bound_plan):
    assert slot_bound_plan.accepted_orders == ()


def test_slot_bound_market_spill_signal_is_flagged_non_actionable(slot_bound_plan):
    spill_signals = [s for s in slot_bound_plan.signals if s.signal_type == "market_spill"]
    assert spill_signals
    assert all(not s.actionable for s in spill_signals)


# -- white-box: rolling baseline, not independent scoring ------------------------


def test_evaluate_candidate_reflects_capacity_already_committed_in_base_fleet_config():
    """The correctness property recommend_expansion's rolling baseline exists for:
    the exact same candidate evaluated against a fleet that already has plenty of
    the type on order must show materially less benefit than evaluating it against
    a genuinely scarce baseline. If evaluate_candidate ignored `base_fleet_config`
    and always solved against a fresh load, these two numbers would be identical.
    """
    scarce_cfg = _single_type_config("A320neo", 2)
    scarce_ladder = run_ladder(2027, 2028, markets=[DEL_BOM_MARKET], fleet_config=scarce_cfg)
    signal = next(s for s in detect_signals(scarce_ladder) if s.signal_type == "fleet_utilisation")
    candidate = size_candidate(
        signal, end_year=2028, fleet_config=scarce_cfg, markets=[DEL_BOM_MARKET]
    )

    scarce_eval = evaluate_candidate(candidate, baseline_ladder=scarce_ladder, base_fleet_config=scarce_cfg)

    generous_cfg = copy.deepcopy(scarce_cfg)
    generous_cfg["orders"].append(
        {
            "type": "A320neo",
            "count": 20,
            "start": date(2027, 1, 1),
            "end": date(2027, 6, 30),
            "confidence": 1.0,
        }
    )
    generous_ladder = run_ladder(2027, 2028, markets=[DEL_BOM_MARKET], fleet_config=generous_cfg)
    generous_eval = evaluate_candidate(
        candidate, baseline_ladder=generous_ladder, base_fleet_config=generous_cfg
    )

    assert generous_eval.npv_usd < scarce_eval.npv_usd
