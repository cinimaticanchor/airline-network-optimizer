"""The time-space fleet-assignment core -- a trimmed first increment.

Validated only on a small type scope (see the module docstring on
`anos.optimize.timespace_assignment` for why: back-of-envelope variable-count sizing
against the full network is a real risk this increment exists to avoid taking on
blind). Same solver-gap tolerance convention as the other extension test files.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from anos.data.fleet_timeline import fleet_on
from anos.data.loaders import load_markets
from anos.optimize.fleet_assignment import solve_network
from anos.optimize.tail_feasibility import check_plan
from anos.optimize.timespace_assignment import solve_network_timespace

TARGET = date(2027, 3, 1)
SCOPE = ["A320neo", "A321neo"]  # the trimmed first-increment scope


@pytest.fixture(scope="module")
def ts_plan():
    return solve_network_timespace(TARGET, timespace_types=SCOPE)


def test_requires_nonempty_timespace_types():
    with pytest.raises(ValueError):
        solve_network_timespace(TARGET, timespace_types=[])


def test_solver_reaches_a_solution(ts_plan):
    assert ts_plan.solver_status in ("OPTIMAL", "FEASIBLE")
    assert ts_plan.markets_served() > 0


def test_timespace_types_recorded_on_the_plan(ts_plan):
    assert ts_plan.timespace_types == sorted(SCOPE)


def test_scheduled_legs_are_populated(ts_plan):
    assert ts_plan.scheduled_legs
    for leg in ts_plan.scheduled_legs:
        assert leg.type_code in SCOPE
        assert leg.direction in ("outbound", "return")
        assert leg.count > 0


def test_outbound_and_return_leg_counts_agree_with_reported_frequency(ts_plan):
    by_market_type_dir: dict[tuple[str, str, str], int] = {}
    for leg in ts_plan.scheduled_legs:
        key = (leg.market_id, leg.type_code, leg.direction)
        by_market_type_dir[key] = by_market_type_dir.get(key, 0) + leg.count

    for mp in ts_plan.market_plans:
        for code, n in mp.frequencies.items():
            if code not in SCOPE:
                continue
            assert by_market_type_dir.get((mp.market_id, code, "outbound"), 0) == n
            assert by_market_type_dir.get((mp.market_id, code, "return"), 0) == n


def test_flow_conservation_never_exceeds_available_fleet_count(ts_plan):
    """Independent recomputation: sum outbound departures crossing the anchor cut,
    per type, can never exceed the floored available airframe count -- the whole
    point of the flow-balance construction."""
    from anos.network.timespace_clock import bucket_of, n_buckets, straddles_anchor

    snapshot = fleet_on(TARGET)
    cfg = {"buckets": 8, "anchor_hour": 3.0}
    n = n_buckets(cfg)
    anchor = bucket_of(cfg["anchor_hour"], cfg)

    for code in SCOPE:
        available = math.floor(snapshot.available.get(code, 0.0))
        in_transit = 0
        for leg in ts_plan.scheduled_legs:
            if leg.type_code != code:
                continue
            if straddles_anchor(leg.departure_bucket, leg.avail_bucket, anchor, n):
                in_transit += leg.count
        assert in_transit <= available


def test_types_outside_scope_still_use_aggregate_hours(ts_plan):
    """A type not in timespace_types must be capacity-checked exactly the way
    fleet_assignment.solve_network does it -- same idiom as
    test_fleet_hours_used_agree_with_the_assignment."""
    from anos.costs.economics import build_economics

    econ = build_economics()
    recomputed: dict[str, float] = {}
    for mp in ts_plan.market_plans:
        for code, n in mp.frequencies.items():
            if code in SCOPE:
                continue
            recomputed[code] = (
                recomputed.get(code, 0.0) + econ[(mp.market_id, code)].aircraft_hours_consumed * n
            )
    for code, hours in recomputed.items():
        assert ts_plan.fleet_used[code] == pytest.approx(hours)
        assert hours <= ts_plan.fleet_available.get(code, 0.0) + 1e-6


def test_shared_invariants_still_hold(ts_plan):
    """Everything fleet_assignment's own per-market loop enforces must still hold,
    since it's duplicated unchanged for this construction."""
    markets = {m.market_id: m for m in load_markets()}
    for mp in ts_plan.market_plans:
        market = markets[mp.market_id]
        assert mp.pax_carried <= mp.seats_offered + 1e-6
        assert mp.total_frequency <= market.max_daily_freq
        assert mp.total_frequency >= market.min_daily_freq
        assert sum(mp.frequencies.values()) == mp.total_frequency


def test_timespace_scoped_types_pass_tail_feasibility_utilisation_check(ts_plan):
    """The aggregate hours cap is deliberately kept active for timespace-scoped
    types too (see the module docstring) precisely so this holds -- a flow-feasible
    plan must also be utilisation-feasible, not just fleet-count-feasible."""
    report = check_plan(ts_plan)
    blocked_types = {f.subject for f in report.blockers if f.category == "fleet capacity"}
    assert not blocked_types & set(SCOPE)


def test_timespace_is_a_strict_refinement_cannot_exceed_aggregate_contribution():
    """Every constraint the aggregate model enforces is still enforced here, plus
    the flow network on top -- the feasible region can only shrink, so the optimum
    can only fall (within solver-gap noise, same idiom as
    test_a_larger_fleet_cannot_make_the_network_worse)."""
    aggregate = solve_network(TARGET)
    timespace = solve_network_timespace(TARGET, timespace_types=SCOPE)
    assert timespace.total_contribution_usd <= aggregate.total_contribution_usd + (
        0.01 * aggregate.total_contribution_usd
    )


def test_solve_network_unaffected_by_importing_timespace_module():
    """Cheap insurance against shared lru_cache config-loader pollution from the new
    data/timespace.yaml loader."""
    plain = solve_network(TARGET)
    assert plain.total_contribution_usd > 0  # importing this test module must not have broken it


def test_timespace_solve_completes_within_budget(ts_plan):
    from anos.config import load_params

    assert ts_plan.solve_seconds < load_params().solver_max_seconds


def test_does_not_mutate_shared_reference_data():
    before = [m.__dict__.copy() for m in load_markets()]
    solve_network_timespace(TARGET, timespace_types=SCOPE)
    after = [m.__dict__.copy() for m in load_markets()]
    assert before == after


# -- Step B regression: types with zero current fleet count --------------------


def test_zero_fleet_count_type_in_scope_does_not_crash():
    """A321XLR has zero aircraft in service before its 2029 order window opens
    (data/fleet.yaml) -- including it in timespace_types must not KeyError, and
    every market's A321XLR frequency must be forced to exactly 0."""
    current = date(2026, 8, 3)
    snapshot = fleet_on(current)
    assert math.floor(snapshot.available.get("A321XLR", 0.0)) == 0  # confirms the premise

    plan = solve_network_timespace(current, timespace_types=[*SCOPE, "A321XLR"])
    for mp in plan.market_plans:
        assert mp.frequencies.get("A321XLR", 0) == 0


# -- Step D: tail-rotation decomposition, wired into the solver -----------------


def test_build_tail_rotations_defaults_to_off(ts_plan):
    assert ts_plan.tail_rotations == []


def test_build_tail_rotations_true_populates_rotations():
    plan = solve_network_timespace(TARGET, timespace_types=SCOPE, build_tail_rotations=True)
    assert plan.tail_rotations
    for rotation in plan.tail_rotations:
        assert rotation.type_code in SCOPE
        assert rotation.tail_count > 0
        assert rotation.legs


def test_ground_occupancy_is_populated_unconditionally(ts_plan):
    """Unlike tail_rotations, ground_occupancy is cheap and always extracted --
    independent sanity bounds on what a raw solver read should look like.

    Note: a station is only guaranteed to appear here if it is *economically
    eligible* for the type (build_economics has an entry), not if it was actually
    *scheduled* -- a station with zero scheduled frequency for a type has no flight
    arcs touching it, so its ground variable is only constrained by a trivial
    self-loop equality (ground[b-1] == ground[b]) and CP-SAT is free to leave it at
    any value in its domain. This is harmless (these variables carry no cost and
    touch no other constraint) but means ground_occupancy can list a station the
    type never actually flies to in this plan."""
    from anos.costs.economics import build_economics

    assert ts_plan.ground_occupancy
    snapshot = fleet_on(TARGET)
    econ = build_economics()
    markets = {m.market_id: m for m in load_markets()}
    eligible_stations: dict[str, set[str]] = {code: set() for code in SCOPE}
    for mid, code in econ:
        if code in SCOPE:
            market = markets[mid]
            eligible_stations[code].update({market.origin, market.destination})

    for (code, station, bucket), count in ts_plan.ground_occupancy.items():
        assert code in SCOPE
        assert count > 0
        assert count <= math.floor(snapshot.available.get(code, 0.0))
        assert 0 <= bucket < 8
        assert station in eligible_stations[code]


def test_tail_rotations_edge_weights_reconcile_with_scheduled_legs():
    """Integration-level cross-check of the same invariant
    tests/test_tail_routing.py proves in isolation: every scheduled leg's count and
    every ground occupancy's count is exactly accounted for across the decomposed
    rotations -- using a real solved plan, not hand-built fixtures."""
    plan = solve_network_timespace(TARGET, timespace_types=SCOPE, build_tail_rotations=True)

    consumed: dict[tuple, int] = {}
    for rotation in plan.tail_rotations:
        for leg in rotation.legs:
            key = (
                rotation.type_code, leg.kind,
                leg.from_station, leg.from_bucket, leg.to_station, leg.to_bucket,
            )
            consumed[key] = consumed.get(key, 0) + rotation.tail_count

    markets = {m.market_id: m for m in load_markets()}
    for leg in plan.scheduled_legs:
        market = markets[leg.market_id]
        if leg.direction == "outbound":
            from_station, to_station = market.origin, market.destination
        else:
            from_station, to_station = market.destination, market.origin
        key = (leg.type_code, "fly", from_station, leg.departure_bucket, to_station, leg.avail_bucket)
        assert consumed.get(key, 0) == leg.count

    for (code, station, bucket), count in plan.ground_occupancy.items():
        key = (code, "ground", station, bucket, station, (bucket + 1) % 8)
        assert consumed.get(key, 0) == count
