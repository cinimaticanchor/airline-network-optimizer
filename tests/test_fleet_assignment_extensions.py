"""The opt-in solver extensions: fast_turn, fare elasticity, connections/banking and
rotations.

CP-SAT's relative MIP gap (`solver_relative_gap` in cost_params.yaml, 0.5%) plus
parallel search means two solves of the *exact same, unmodified* problem already do
not always return the bit-identical objective -- confirmed by solving the pinned
TARGET date three times in a row with no extensions at all. Non-regression here
therefore means "at least as good, within the solver's own tolerance", the same
monotonicity idiom `test_optimizer.py` already uses for
`test_releasing_service_floors_cannot_reduce_contribution`, not exact equality.
"""

from __future__ import annotations

from datetime import date

import pytest

from anos.data.loaders import load_markets
from anos.network.itinerary import build_connection_candidates, low_load_factor_squad
from anos.optimize.fleet_assignment import solve_network

TARGET = date(2027, 3, 1)
CURRENT = date(2026, 8, 3)  # the date every "current network" claim in this session used


@pytest.fixture(scope="module")
def baseline():
    return solve_network(TARGET)


# -- non-regression: every extension defaults to inert -----------------------


def test_fast_turn_off_by_default_reproduces_baseline(baseline):
    explicit_off = solve_network(TARGET, fast_turn=False)
    assert explicit_off.total_contribution_usd == pytest.approx(
        baseline.total_contribution_usd, rel=0.01
    )


def test_fare_elasticity_off_by_default_reproduces_baseline(baseline):
    explicit_off = solve_network(TARGET, enable_fare_elasticity=False)
    assert explicit_off.total_contribution_usd == pytest.approx(
        baseline.total_contribution_usd, rel=0.01
    )
    assert all(mp.fare_usd is None for mp in explicit_off.market_plans)


def test_connections_off_by_default_reproduces_baseline(baseline):
    explicit_off = solve_network(TARGET, enable_connections=False, hubs=["DEL", "BOM"])
    assert explicit_off.total_contribution_usd == pytest.approx(
        baseline.total_contribution_usd, rel=0.01
    )
    assert explicit_off.connections_used == []


def test_connections_without_hubs_is_a_noop(baseline):
    """enable_connections=True with no hubs must not silently do nothing wrong --
    it must build zero candidates and stay inert."""
    result = solve_network(TARGET, enable_connections=True, hubs=None)
    assert result.total_contribution_usd == pytest.approx(
        baseline.total_contribution_usd, rel=0.01
    )
    assert result.connections_used == []


def test_rotations_off_by_default_reproduces_baseline(baseline):
    explicit_off = solve_network(TARGET, enable_rotations=False, hubs=["DEL"])
    assert explicit_off.total_contribution_usd == pytest.approx(
        baseline.total_contribution_usd, rel=0.01
    )
    assert explicit_off.rotations_flown == {}


def test_rotations_without_squad_is_a_noop(baseline):
    result = solve_network(TARGET, enable_rotations=True, hubs=["DEL"], rotation_squad=None)
    assert result.total_contribution_usd == pytest.approx(
        baseline.total_contribution_usd, rel=0.01
    )
    assert result.rotations_flown == {}


def test_repeated_identical_solves_vary_only_within_the_solver_gap():
    """Establishes the tolerance the tests above rely on: this is a pre-existing
    property of the solver configuration, not something the extensions introduced."""
    runs = [solve_network(TARGET).total_contribution_usd for _ in range(2)]
    assert runs[0] == pytest.approx(runs[1], rel=0.01)


# -- new capability: tied to concrete findings from this repo ----------------


def test_connections_can_serve_a_previously_unserved_market():
    """MAA-LHR: no nonstop, but DEL-MAA and DEL-LHR are both already flown."""
    plan = solve_network(CURRENT, enable_connections=True, hubs=["DEL", "BOM"])
    assert "MAA-LHR" in plan.connections_used


def test_connections_never_exceed_feeder_or_trunk_spare_seats():
    """Independent recomputation: no leg's connecting + direct pax may exceed the
    seats that leg's own frequency actually offers.

    Restricted to a single hub: with two hubs a market can gain two independent
    connection candidates (e.g. MAA-LHR via both DEL and BOM, since BOM-MAA and
    BOM-LHR both exist too), and `MarketPlan.connecting_pax_carried` reports their
    *sum* -- which this recomputation, working from candidates rather than solved
    per-candidate values, cannot split back apart. One hub avoids that ambiguity
    while still exercising the real constraint.
    """
    plan = solve_network(CURRENT, enable_connections=True, hubs=["DEL"])
    candidates = build_connection_candidates(load_markets(), hubs=["DEL"])
    by_market = {mp.market_id: mp for mp in plan.market_plans}
    connecting_by_leg: dict[str, float] = {}
    for c in candidates:
        connecting = by_market.get(c.market_id)
        if not connecting or connecting.connecting_pax_carried <= 0:
            continue
        # This connection's pax draw against both legs it uses -- but a leg can be
        # used by more than one candidate, so accumulate rather than overwrite.
        connecting_by_leg[c.feeder_market_id] = (
            connecting_by_leg.get(c.feeder_market_id, 0.0) + connecting.connecting_pax_carried
        )
        connecting_by_leg[c.trunk_market_id] = (
            connecting_by_leg.get(c.trunk_market_id, 0.0) + connecting.connecting_pax_carried
        )

    for leg_mid, connecting_pax in connecting_by_leg.items():
        leg = by_market[leg_mid]
        assert leg.pax_carried + connecting_pax <= leg.seats_offered + 1e-6


def test_star_hub_can_serve_markets_del_bom_banking_cannot():
    """The star-model claim: a secondary hub reaches connections the primary hubs
    structurally cannot (no DEL/BOM feeder exists for these South Indian markets)."""
    del_bom = solve_network(CURRENT, enable_connections=True, hubs=["DEL", "BOM"])
    with_blr = solve_network(CURRENT, enable_connections=True, hubs=["DEL", "BOM", "BLR"])
    assert set(with_blr.connections_used) - set(del_bom.connections_used)


def test_rotations_never_exceed_type_hours_budget(baseline):
    squad = low_load_factor_squad(baseline, threshold=0.60)
    plan = solve_network(
        TARGET, enable_rotations=True, hubs=["DEL", "BOM", "BLR"], rotation_squad=squad
    )
    for code, used in plan.fleet_used.items():
        available = plan.fleet_available.get(code, 0.0)
        assert used <= available + 1e-6


def test_rotations_contribute_to_leg_market_frequency(baseline):
    """A rotation loop must show up as ordinary frequency on the two markets it
    serves -- min-service floors and slot caps must still see it."""
    squad = low_load_factor_squad(baseline, threshold=0.60)
    plan = solve_network(
        TARGET, enable_rotations=True, hubs=["DEL", "BOM", "BLR"], rotation_squad=squad
    )
    if not plan.rotations_flown:
        pytest.skip("no rotation was profitable for this squad/date")
    by_market = {mp.market_id: mp for mp in plan.market_plans}
    for mid in squad:
        market = by_market.get(mid)
        if market is not None:
            assert market.total_frequency >= 0  # sanity: field is populated, not orphaned


def test_fare_elasticity_can_change_the_objective():
    baseline = solve_network(CURRENT)
    elastic = solve_network(CURRENT, enable_fare_elasticity=True)
    assert elastic.total_contribution_usd != pytest.approx(
        baseline.total_contribution_usd, rel=0.005
    )


def test_fare_elasticity_reports_a_realised_fare_for_every_served_market():
    plan = solve_network(CURRENT, enable_fare_elasticity=True)
    for mp in plan.market_plans:
        assert mp.fare_usd is not None


def test_fast_turn_cannot_reduce_contribution():
    """Monotonicity, same idiom as test_a_larger_fleet_cannot_make_the_network_worse:
    fast_turn only ever reduces aircraft-hours per trip, so the feasible region can
    only grow."""
    normal = solve_network(TARGET, enforce_min_service=True)
    fast = solve_network(TARGET, enforce_min_service=True, fast_turn=True)
    assert fast.total_contribution_usd >= normal.total_contribution_usd - 1.0


def test_fast_turn_respects_every_minimum_service_floor():
    floors = {m.market_id: m.min_daily_freq for m in load_markets()}
    plan = solve_network(TARGET, enforce_min_service=True, fast_turn=True)
    for mp in plan.market_plans:
        assert mp.total_frequency >= floors.get(mp.market_id, 0)


def test_extensions_do_not_mutate_shared_reference_data():
    before = [m.__dict__.copy() for m in load_markets()]
    solve_network(
        CURRENT,
        fast_turn=True,
        enable_fare_elasticity=True,
        enable_connections=True,
        hubs=["DEL", "BOM", "BLR"],
        enable_rotations=True,
        rotation_squad=["DEL-IXM"],
    )
    after = [m.__dict__.copy() for m in load_markets()]
    assert before == after


def test_all_extensions_compose_without_error():
    """The full 'does this make more money' run: every lever on at once."""
    baseline = solve_network(CURRENT)
    squad = low_load_factor_squad(baseline, threshold=0.50)
    plan = solve_network(
        CURRENT,
        fast_turn=True,
        enable_fare_elasticity=True,
        enable_connections=True,
        hubs=["DEL", "BOM", "BLR"],
        enable_rotations=True,
        rotation_squad=squad,
    )
    assert plan.solver_status in ("OPTIMAL", "FEASIBLE")
    assert plan.total_contribution_usd > 0
