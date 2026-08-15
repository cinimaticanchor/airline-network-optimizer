"""The optimiser must respect every constraint it claims to model.

A solver that quietly violates a constraint produces a plan that looks better than
reality, which is the most expensive failure mode this system has.
"""

from __future__ import annotations

from datetime import date

import pytest

from anos.costs.economics import build_economics
from anos.data.fleet_timeline import daily_aircraft_hours, fleet_on
from anos.data.loaders import load_aircraft_types, load_airports, load_markets
from anos.forecast.demand import effective_demand
from anos.optimize.fleet_assignment import InfeasibleNetworkError, solve_network

TARGET = date(2027, 3, 1)


@pytest.fixture(scope="module")
def plan():
    return solve_network(TARGET)


def test_solver_reaches_a_solution(plan):
    assert plan.solver_status in ("OPTIMAL", "FEASIBLE")
    assert plan.markets_served() > 0


def test_plan_is_profitable_overall(plan):
    assert plan.total_contribution_usd > 0
    assert plan.total_revenue_usd > plan.total_cost_usd


def test_fleet_capacity_is_never_exceeded(plan):
    """The hard constraint: you cannot fly more hours than you have aircraft for."""
    for code, used in plan.fleet_used.items():
        available = plan.fleet_available.get(code, 0.0)
        assert used <= available + 1e-6, f"{code} over-utilised: {used} > {available}"


def test_only_eligible_aircraft_are_assigned(plan):
    """No aircraft is assigned to a sector it cannot physically fly."""
    econ = build_economics()
    for mp in plan.market_plans:
        for code in mp.frequencies:
            assert (mp.market_id, code) in econ, f"{code} assigned to ineligible {mp.market_id}"


def test_widebodies_never_sent_to_incapable_airports(plan):
    types = load_aircraft_types()
    ports = load_airports()
    markets = {m.market_id: m for m in load_markets()}

    for mp in plan.market_plans:
        for code in mp.frequencies:
            if not types[code].is_widebody:
                continue
            market = markets[mp.market_id]
            assert ports[market.origin].widebody_capable
            assert ports[market.destination].widebody_capable


def test_passengers_never_exceed_seats_offered(plan):
    for mp in plan.market_plans:
        assert mp.pax_carried <= mp.seats_offered + 1e-6
        assert 0.0 <= mp.load_factor <= 1.0 + 1e-9


def test_passengers_never_exceed_demand(plan):
    """You cannot carry more people than want to travel."""
    markets = {m.market_id: m for m in load_markets()}
    for mp in plan.market_plans:
        if mp.total_frequency == 0:
            continue
        demand = effective_demand(markets[mp.market_id], mp.total_frequency, plan.month)
        assert mp.pax_carried <= demand + 1.0


def test_frequency_bounds_are_respected(plan):
    markets = {m.market_id: m for m in load_markets()}
    for mp in plan.market_plans:
        market = markets[mp.market_id]
        assert mp.total_frequency <= market.max_daily_freq
        assert mp.total_frequency >= market.min_daily_freq


def test_slot_allocations_are_respected(plan):
    ports = load_airports()
    markets = {m.market_id: m for m in load_markets()}
    usage: dict[str, int] = {}
    for mp in plan.market_plans:
        market = markets[mp.market_id]
        for iata in (market.origin, market.destination):
            usage[iata] = usage.get(iata, 0) + mp.total_frequency

    for iata, used in usage.items():
        assert used <= ports[iata].daily_slot_cap, f"{iata} over slot allocation"


def test_frequencies_sum_to_the_market_total(plan):
    for mp in plan.market_plans:
        assert sum(mp.frequencies.values()) == mp.total_frequency


def test_releasing_service_floors_cannot_reduce_contribution():
    """Dropping a constraint can only widen the feasible set, so the objective must
    not fall. A drop here means the model is mis-specified."""
    constrained = solve_network(TARGET)
    free = solve_network(TARGET, enforce_min_service=False)
    assert free.total_contribution_usd >= constrained.total_contribution_usd - 1.0


def test_a_larger_fleet_cannot_make_the_network_worse():
    """Monotonicity in capacity -- another way the model would be mis-specified."""
    lean = solve_network(TARGET, fleet=fleet_on(TARGET, probabilistic=True))
    full = solve_network(TARGET, fleet=fleet_on(TARGET, probabilistic=False))
    assert full.total_contribution_usd >= lean.total_contribution_usd - 1.0


def test_seasonality_changes_the_answer():
    """Solving for a peak month should not return the trough month's network."""
    winter = solve_network(TARGET, month=1)
    summer = solve_network(TARGET, month=7)
    assert winter.total_contribution_usd != summer.total_contribution_usd


def test_an_empty_fleet_is_reported_as_infeasible():
    """With no aircraft and mandatory service floors, say so clearly."""
    empty = fleet_on(TARGET)
    empty.available = {code: 0.0 for code in empty.available}
    with pytest.raises(InfeasibleNetworkError):
        solve_network(TARGET, fleet=empty)


def test_reported_totals_agree_with_the_per_market_figures(plan):
    assert plan.total_revenue_usd == pytest.approx(
        sum(mp.revenue_usd for mp in plan.market_plans)
    )
    assert plan.total_contribution_usd == pytest.approx(
        plan.total_revenue_usd - plan.total_cost_usd
    )


def test_fleet_hours_used_agree_with_the_assignment(plan):
    """Recompute utilisation independently of the solver's own accounting."""
    econ = build_economics()
    recomputed: dict[str, float] = {}
    for mp in plan.market_plans:
        for code, n in mp.frequencies.items():
            recomputed[code] = (
                recomputed.get(code, 0.0) + econ[(mp.market_id, code)].aircraft_hours_consumed * n
            )
    for code, hours in recomputed.items():
        assert plan.fleet_used[code] == pytest.approx(hours)


def test_available_hours_match_the_fleet_snapshot(plan):
    expected = daily_aircraft_hours(fleet_on(TARGET))
    for code, hours in expected.items():
        assert plan.fleet_available.get(code, 0.0) == pytest.approx(hours)
