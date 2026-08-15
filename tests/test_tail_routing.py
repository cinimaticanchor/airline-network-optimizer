"""Tail-rotation decomposition -- pure graph algorithm, no CP-SAT anywhere in this
file. Each case targets a specific, real bug class, not a happy-path smoke test;
see `anos.optimize.tail_routing`'s module docstring for the correctness argument
these tests are checking.
"""

from __future__ import annotations

import copy

from anos.models import LegEconomics, ScheduledLegGroup
from anos.optimize.tail_routing import (
    _Edge,
    _extract_cycles,
    decompose_flow_into_rotations,
)
from tests.test_demand import make_market


def make_econ(market_id: str, type_code: str, **overrides) -> LegEconomics:
    base = dict(
        market_id=market_id,
        type_code=type_code,
        distance_km=1000.0,
        block_hours_round_trip=4.0,
        aircraft_hours_consumed=4.0,
        seats_round_trip=180,
        cost_per_round_trip_usd=10000.0,
        eligible=True,
    )
    base.update(overrides)
    return LegEconomics(**base)


# -- decompose_flow_into_rotations (public function) -----------------------------


def test_all_ground_station_decomposes_to_a_single_no_fly_rotation():
    """A type that never flies, only sits, must still produce a rotation -- not be
    silently dropped for lacking a fly-hop."""
    buckets = 4
    ground_occupancy = {("X", "A", b): 3 for b in range(buckets)}

    rotations = decompose_flow_into_rotations(
        scheduled_legs=[],
        ground_occupancy=ground_occupancy,
        markets=[],
        economics={},
        timespace_types=["X"],
        buckets=buckets,
    )

    assert len(rotations) == 1
    rotation = rotations[0]
    assert rotation.type_code == "X"
    assert rotation.tail_count == 3
    assert rotation.daily_flight_hours == 0.0
    assert all(leg.kind == "ground" for leg in rotation.legs)
    assert len(rotation.legs) == buckets


def test_two_markets_sharing_a_route_keep_separate_attribution():
    """Two different markets whose legs happen to share the exact same
    (from_node, to_node) -- a genuine parallel-edge case that would silently merge
    under a dict[(from,to),weight] representation -- must stay attributed to the
    correct market_id, with per-market weight fully conserved."""
    buckets = 2
    m1 = make_market(market_id="M1", origin="A", destination="B", max_daily_freq=10)
    m2 = make_market(market_id="M2", origin="A", destination="B", max_daily_freq=10)

    scheduled_legs = [
        ScheduledLegGroup("M1", "X", "outbound", departure_bucket=0, avail_bucket=1, count=2),
        ScheduledLegGroup("M2", "X", "outbound", departure_bucket=0, avail_bucket=1, count=5),
        ScheduledLegGroup("M1", "X", "return", departure_bucket=1, avail_bucket=0, count=2),
        ScheduledLegGroup("M2", "X", "return", departure_bucket=1, avail_bucket=0, count=5),
    ]
    economics = {
        ("M1", "X"): make_econ("M1", "X", aircraft_hours_consumed=4.0),  # 2.0h one-way
        ("M2", "X"): make_econ("M2", "X", aircraft_hours_consumed=6.0),  # 3.0h one-way
    }

    rotations = decompose_flow_into_rotations(
        scheduled_legs=scheduled_legs,
        ground_occupancy={},
        markets=[m1, m2],
        economics=economics,
        timespace_types=["X"],
        buckets=buckets,
    )

    by_market = {}
    for r in rotations:
        market_ids = {leg.market_id for leg in r.legs}
        assert len(market_ids) == 1, "a single rotation must not mix two markets"
        by_market[market_ids.pop()] = r

    assert by_market["M1"].tail_count == 2
    assert by_market["M2"].tail_count == 5
    assert by_market["M1"].daily_flight_hours == 4.0  # 2 legs x 2.0h
    assert by_market["M2"].daily_flight_hours == 6.0  # 2 legs x 3.0h
    # every fly-leg's to_station must match what its own market+direction implies --
    # never borrowed from the other market sharing the same route.
    for r, market in ((by_market["M1"], m1), (by_market["M2"], m2)):
        for leg in r.legs:
            expected_to = market.destination if leg.direction == "outbound" else market.origin
            assert leg.to_station == expected_to


def test_edge_weight_conservation_across_fly_and_ground():
    """For every distinct input edge, the total tail_count attributed to it across
    all output rotations equals its original weight exactly -- the unconditional
    invariant (not sum(tail_count) == fleet_count, which only holds for single-lap
    cycles -- see the module docstring)."""
    buckets = 4
    market = make_market(market_id="P-Q", origin="P", destination="Q", max_daily_freq=10)
    scheduled_legs = [
        ScheduledLegGroup("P-Q", "X", "outbound", departure_bucket=0, avail_bucket=1, count=4),
        ScheduledLegGroup("P-Q", "X", "return", departure_bucket=2, avail_bucket=3, count=4),
    ]
    ground_occupancy = {
        ("X", "Q", 1): 4,  # sits at Q from bucket 1 to 2
        ("X", "P", 3): 4,  # sits at P from bucket 3 to 0 (wraps)
    }
    economics = {("P-Q", "X"): make_econ("P-Q", "X", aircraft_hours_consumed=2.0)}

    rotations = decompose_flow_into_rotations(
        scheduled_legs=scheduled_legs,
        ground_occupancy=ground_occupancy,
        markets=[market],
        economics=economics,
        timespace_types=["X"],
        buckets=buckets,
    )

    consumed: dict[tuple, int] = {}
    for r in rotations:
        for leg in r.legs:
            key = (leg.kind, leg.from_station, leg.from_bucket, leg.to_station, leg.to_bucket)
            consumed[key] = consumed.get(key, 0) + r.tail_count

    assert consumed[("fly", "P", 0, "Q", 1)] == 4
    assert consumed[("fly", "Q", 2, "P", 3)] == 4
    assert consumed[("ground", "Q", 1, "Q", 2)] == 4
    assert consumed[("ground", "P", 3, "P", 0)] == 4


def test_empty_input_returns_no_rotations():
    assert decompose_flow_into_rotations([], {}, [], {}, ["X"], 4) == []


def test_type_with_zero_flow_is_skipped_not_errored():
    """A timespace type with nothing scheduled for it (e.g. zero fleet count on the
    target date) must be silently absent from the output, not raise."""
    rotations = decompose_flow_into_rotations(
        scheduled_legs=[], ground_occupancy={}, markets=[], economics={},
        timespace_types=["X", "Y"], buckets=4,
    )
    assert rotations == []


def test_deep_copy_isolation_inputs_unchanged():
    buckets = 2
    m1 = make_market(market_id="M1", origin="A", destination="B", max_daily_freq=10)
    scheduled_legs = [
        ScheduledLegGroup("M1", "X", "outbound", departure_bucket=0, avail_bucket=1, count=3),
        ScheduledLegGroup("M1", "X", "return", departure_bucket=1, avail_bucket=0, count=3),
    ]
    ground_occupancy = {("X", "C", 0): 2, ("X", "C", 1): 2}
    economics = {("M1", "X"): make_econ("M1", "X")}

    legs_before = copy.deepcopy(scheduled_legs)
    ground_before = copy.deepcopy(ground_occupancy)

    decompose_flow_into_rotations(
        scheduled_legs=scheduled_legs,
        ground_occupancy=ground_occupancy,
        markets=[m1],
        economics=economics,
        timespace_types=["X"],
        buckets=buckets,
    )

    assert scheduled_legs == legs_before
    assert ground_occupancy == ground_before


def test_determinism_identical_input_gives_identical_output():
    buckets = 2
    m1 = make_market(market_id="M1", origin="A", destination="B", max_daily_freq=10)
    m2 = make_market(market_id="M2", origin="A", destination="B", max_daily_freq=10)
    scheduled_legs = [
        ScheduledLegGroup("M1", "X", "outbound", departure_bucket=0, avail_bucket=1, count=2),
        ScheduledLegGroup("M2", "X", "outbound", departure_bucket=0, avail_bucket=1, count=5),
        ScheduledLegGroup("M1", "X", "return", departure_bucket=1, avail_bucket=0, count=2),
        ScheduledLegGroup("M2", "X", "return", departure_bucket=1, avail_bucket=0, count=5),
    ]
    economics = {
        ("M1", "X"): make_econ("M1", "X", aircraft_hours_consumed=4.0),
        ("M2", "X"): make_econ("M2", "X", aircraft_hours_consumed=6.0),
    }

    r1 = decompose_flow_into_rotations(scheduled_legs, {}, [m1, m2], economics, ["X"], buckets)
    r2 = decompose_flow_into_rotations(scheduled_legs, {}, [m1, m2], economics, ["X"], buckets)

    assert r1 == r2


# -- _extract_cycles (generic, domain-agnostic graph algorithm) -------------------


def test_extract_cycles_stem_before_cycle_is_correctly_separated():
    """The single most important correctness case: a walk that traverses a 'stem'
    edge before reaching a node that repeats *later*, not at the walk's own start.
    A buggy implementation that always treats the whole walked path as the cycle
    would either corrupt conservation or crash on a later iteration -- see the
    module docstring's worked trace of this exact graph.

    Topology: a 2-edge pendant (L<->J) hangs off a 3-edge loop (J-K-M-J), sharing
    junction J. Node names are chosen so the deterministic tie-break (alphabetical
    on to_node) is forced to walk L->J->K->M->J first -- discovering the inner
    3-edge loop [J->K, K->M, M->J] while correctly leaving the L->J stem edge
    untouched for a later iteration, which then pairs it with J->L.
    """
    edges = [
        _Edge("ground", ("L", 0), ("J", 0), 10, None, None, 0.0),  # stem: L -> J
        _Edge("ground", ("J", 0), ("L", 0), 10, None, None, 0.0),  # J -> L
        _Edge("ground", ("J", 0), ("K", 0), 10, None, None, 0.0),  # J -> K
        _Edge("ground", ("K", 0), ("M", 0), 10, None, None, 0.0),  # K -> M
        _Edge("ground", ("M", 0), ("J", 0), 10, None, None, 0.0),  # M -> J
    ]

    cycles = _extract_cycles(edges)

    assert len(cycles) == 2

    # Every input edge's weight is fully and exactly accounted for.
    consumed: dict[tuple, int] = {}
    for cycle_edges, weight in cycles:
        for e in cycle_edges:
            consumed[(e.from_node, e.to_node)] = consumed.get((e.from_node, e.to_node), 0) + weight
    expected = {(e.from_node, e.to_node): e.weight for e in edges}
    assert consumed == expected

    # The pendant and the inner loop come out as two SEPARATE cycles, not merged --
    # direct proof the stem was excluded from the inner loop's cycle.
    node_sets = [frozenset(e.from_node for e in cycle_edges) for cycle_edges, _ in cycles]
    assert frozenset({("L", 0), ("J", 0)}) in node_sets
    assert frozenset({("J", 0), ("K", 0), ("M", 0)}) in node_sets


def test_extract_cycles_simple_two_node_loop():
    edges = [
        _Edge("ground", ("A", 0), ("B", 0), 5, None, None, 0.0),
        _Edge("ground", ("B", 0), ("A", 0), 5, None, None, 0.0),
    ]
    cycles = _extract_cycles(edges)
    assert len(cycles) == 1
    cycle_edges, weight = cycles[0]
    assert weight == 5
    assert len(cycle_edges) == 2


def test_extract_cycles_empty_graph():
    assert _extract_cycles([]) == []
