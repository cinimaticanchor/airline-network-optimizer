"""Carbon/SAF cost, demand recapture, and interline -- the three Phase A extensions.

Same solver-gap tolerance convention as `test_fleet_assignment_extensions.py`:
`pytest.approx(rel=0.01)` for solved-objective comparisons, since CP-SAT's own 0.5%
relative gap plus parallel search already makes repeated solves of the identical
problem vary slightly (see that file's module docstring).
"""

from __future__ import annotations

from datetime import date

import pytest

from anos.costs.economics import leg_economics
from anos.data.loaders import load_aircraft_types, load_airports, load_markets
from anos.optimize.fleet_assignment import solve_network
from tests.test_demand import make_market

TARGET = date(2027, 3, 1)
CURRENT = date(2026, 8, 3)


@pytest.fixture(scope="module")
def baseline():
    return solve_network(TARGET)


# -- A1: carbon / SAF -----------------------------------------------------------


def test_carbon_priced_off_by_default_reproduces_baseline(baseline):
    explicit_off = solve_network(TARGET, carbon_priced=False)
    assert explicit_off.total_contribution_usd == pytest.approx(
        baseline.total_contribution_usd, rel=0.01
    )


def test_carbon_cost_only_applies_to_eu_uk_touching_legs():
    types = load_aircraft_types()
    ports = load_airports()
    domestic = make_market(market_id="DEL-BOM", origin="DEL", destination="BOM")
    eu_route = make_market(
        market_id="DEL-LHR", origin="DEL", destination="LHR",
        region="longhaul_europe", max_daily_freq=3,
    )

    domestic_priced = leg_economics(domestic, types["A320neo"], airports=ports, carbon_priced=True)
    domestic_unpriced = leg_economics(domestic, types["A320neo"], airports=ports, carbon_priced=False)
    assert domestic_priced.cost_per_round_trip_usd == pytest.approx(
        domestic_unpriced.cost_per_round_trip_usd
    )

    eu_priced = leg_economics(eu_route, types["B787-9"], airports=ports, carbon_priced=True)
    eu_unpriced = leg_economics(eu_route, types["B787-9"], airports=ports, carbon_priced=False)
    assert eu_priced.cost_per_round_trip_usd > eu_unpriced.cost_per_round_trip_usd


def test_carbon_priced_can_only_reduce_or_hold_contribution():
    """Adding a real cost to a subset of legs can only shrink or hold the feasible
    optimum, never raise it -- same monotonicity idiom as
    test_a_larger_fleet_cannot_make_the_network_worse."""
    normal = solve_network(TARGET)
    priced = solve_network(TARGET, carbon_priced=True)
    assert priced.total_contribution_usd <= normal.total_contribution_usd + 1.0


# -- A2: demand recapture ---------------------------------------------------------


def test_recapture_off_by_default_reproduces_baseline(baseline):
    explicit_off = solve_network(TARGET, enable_recapture=False)
    assert explicit_off.total_contribution_usd == pytest.approx(
        baseline.total_contribution_usd, rel=0.01
    )


def test_recapture_without_connections_is_a_noop(baseline):
    """enable_recapture requires enable_connections; alone it must not do anything."""
    result = solve_network(TARGET, enable_recapture=True)
    assert result.total_contribution_usd == pytest.approx(
        baseline.total_contribution_usd, rel=0.01
    )
    assert result.connections_used == []


def test_recapture_cannot_reduce_contribution_versus_connections_alone():
    """Recapture only widens connect_pax's upper bound -- the feasible region can
    only grow, so the optimum can only rise (within solver-gap noise)."""
    connections_only = solve_network(CURRENT, enable_connections=True, hubs=["DEL", "BOM"])
    with_recapture = solve_network(
        CURRENT, enable_connections=True, enable_recapture=True, hubs=["DEL", "BOM"]
    )
    assert with_recapture.total_contribution_usd >= connections_only.total_contribution_usd - (
        0.01 * connections_only.total_contribution_usd
    )


# -- A3: interline ----------------------------------------------------------------


def test_interline_off_by_default_reproduces_baseline(baseline):
    explicit_off = solve_network(TARGET, enable_interline=False)
    assert explicit_off.total_contribution_usd == pytest.approx(
        baseline.total_contribution_usd, rel=0.01
    )
    assert explicit_off.interline_used == []


def test_markets_csv_flags_exactly_the_structurally_unbankable_markets():
    """The concrete list anos compare surfaced last session: no domestic feeder
    exists for these at any hub, so banking cannot help them."""
    flagged = {m.market_id for m in load_markets() if m.interline_available}
    assert flagged == {"DEL-BHX", "DEL-CPH", "DEL-MXP", "DEL-SYD", "DEL-YVR", "DEL-MEL"}


def test_interline_enabled_serves_flagged_markets():
    plan = solve_network(CURRENT, enable_interline=True)
    assert set(plan.interline_used) <= {
        "DEL-BHX", "DEL-CPH", "DEL-MXP", "DEL-SYD", "DEL-YVR", "DEL-MEL",
    }
    assert plan.interline_used  # at least one should be worth selling at the stated prorate


def test_interline_never_flags_a_market_without_the_csv_flag():
    plan = solve_network(CURRENT, enable_interline=True)
    flagged = {m.market_id for m in load_markets() if m.interline_available}
    assert set(plan.interline_used) <= flagged


def test_interline_cannot_reduce_contribution():
    normal = solve_network(CURRENT)
    interlined = solve_network(CURRENT, enable_interline=True)
    assert interlined.total_contribution_usd >= normal.total_contribution_usd - 1.0


# -- composition --------------------------------------------------------------------


def test_all_phase_a_extensions_compose_without_error():
    plan = solve_network(
        CURRENT,
        carbon_priced=True,
        enable_connections=True,
        enable_recapture=True,
        enable_interline=True,
        hubs=["DEL", "BOM"],
    )
    assert plan.solver_status in ("OPTIMAL", "FEASIBLE")
    assert plan.total_contribution_usd > 0


def test_extensions_do_not_mutate_shared_reference_data():
    before = [m.__dict__.copy() for m in load_markets()]
    solve_network(
        CURRENT,
        carbon_priced=True,
        enable_connections=True,
        enable_recapture=True,
        enable_interline=True,
        hubs=["DEL", "BOM"],
    )
    after = [m.__dict__.copy() for m in load_markets()]
    assert before == after
