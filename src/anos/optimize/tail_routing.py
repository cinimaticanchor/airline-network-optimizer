"""Tail-rotation decomposition: turning an aggregate time-space flow into concrete
example rotations.

`anos.optimize.timespace_assignment` solves for an aggregate *circulation* per
aircraft type -- how many round-trip-direction departures happen in each time
bucket, and how many aircraft sit on the ground in each bucket -- but never says
which physical airframe flies which sequence. This module turns that circulation
into a small number of distinct *rotation patterns*, each shared by `tail_count`
interchangeable synthetic tails (see `anos.models.TailRotation`'s docstring for why
naming individual fake tail IDs would be false precision here).

**The algorithm is flow decomposition, not a fresh optimisation.** The time-space
core's flow-balance constraint at every `(type, station, bucket)` node is exactly
`ground[t,s,b-1] + arrivals == ground[t,s,b] + departures` -- i.e. flight arcs *and*
ground/wait arcs together have zero net divergence everywhere, which is precisely
the definition of a circulation. A circulation can always be decomposed into a
weighted sum of simple directed cycles (the standard network-flow decomposition
theorem): repeatedly walk from any positive-weight edge, following any
positive-weight outgoing edge from the current node, until a node repeats; the
cycle is the suffix starting at that node's *first* occurrence on the walk -- any
"stem" walked before reaching it is left at full weight and picked up by a later
walk, not discarded. Subtract the cycle's bottleneck weight (its smallest edge
weight) from every edge in it and repeat until all weights are zero. This
terminates within `|stations for the type| x buckets` iterations (a finite node
space) and is domain-agnostic -- `_extract_cycles` never looks at markets or
aircraft, only at a generic weighted directed multigraph, so it is exactly as
testable with synthetic non-airline edges as with real ones.

Dropping ground/wait edges and decomposing flight edges alone would NOT work: two
flight arcs are only connected if a ground arc bridges the gap between one
aircraft's landing bucket and its next departure bucket at the same station. The
combined graph is what the flow-balance constraint was proven over; a decomposition
of flight edges alone has no basis in that proof.

Candidate-edge selection (both which edge starts a new extraction and which
outgoing edge continues a walk) uses a fixed deterministic sort key, so identical
inputs always produce byte-identical output -- this is not incidental to how a dict
or set happens to iterate.

**KNOWN LIMITATIONS**, stated explicitly so they are not mistaken for oversights:
1. This does **not** prove multi-day A/C/D-check due-date feasibility.
   `data/fleet.yaml` has no per-tail accumulated-flight-hours history, and even
   with realistic per-type check intervals (`data/aircraft_types.yaml`), a single
   representative day cannot show when a 500-800-flight-hour A-check threshold is
   actually crossed -- only a same-day rotation pattern, plus (at the reporting
   layer) a steady-state "~N days between A-checks at this daily rate"
   extrapolation, both of which are informative but not a due-date proof.
2. The decomposition is **one valid realization**, not unique. A given aggregate
   flow can validly decompose into different sets of cycles; this is a
   deterministic, reproducible choice, not a claim about which physical airframe
   really flies what.
3. No maintenance-base, gate, or crew data exists anywhere in this codebase to
   validate a rotation against (e.g. "does it ever return to a station that can
   perform an A-check") -- rotations are reported as-is.
4. A cycle can legitimately span more than one lap of the bucket clock before
   closing (its length is bounded by node count, not by `buckets`), so
   `sum(tail_count over all cycles of a type)` is only guaranteed to equal that
   type's fleet count for single-lap cycles -- the unconditional invariant is
   per-edge weight conservation (see the test suite).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from anos.models import (
    LegEconomics,
    Market,
    ScheduledLegGroup,
    TailLegAssignment,
    TailRotation,
)

Node = tuple[str, int]  # (station, bucket)


class _Edge(NamedTuple):
    """One directed, weighted arc in a type's decomposition graph."""

    kind: Literal["fly", "ground"]
    from_node: Node
    to_node: Node
    weight: int
    market_id: str | None
    direction: Literal["outbound", "return"] | None
    flight_hours: float


def _edges_for_type(
    type_code: str,
    scheduled_legs: list[ScheduledLegGroup],
    ground_occupancy: dict[tuple[str, str, int], int],
    markets_by_id: dict[str, Market],
    economics: dict[tuple[str, str], LegEconomics],
    buckets: int,
) -> list[_Edge]:
    """Domain mapping: turn one type's scheduled legs and ground occupancy into a
    weighted directed multigraph. Kept separate from `_extract_cycles` so a mapping
    bug (wrong station for a direction) and a graph-algorithm bug are distinguishable
    failure modes.
    """
    edges: list[_Edge] = []
    for leg in scheduled_legs:
        if leg.type_code != type_code or leg.count <= 0:
            continue
        market = markets_by_id[leg.market_id]
        if leg.direction == "outbound":
            from_station, to_station = market.origin, market.destination
        else:
            from_station, to_station = market.destination, market.origin
        flight_hours = economics[(leg.market_id, type_code)].aircraft_hours_consumed / 2.0
        edges.append(
            _Edge(
                kind="fly",
                from_node=(from_station, leg.departure_bucket),
                to_node=(to_station, leg.avail_bucket),
                weight=leg.count,
                market_id=leg.market_id,
                direction=leg.direction,
                flight_hours=flight_hours,
            )
        )

    for (t, station, bucket), weight in sorted(ground_occupancy.items()):
        if t != type_code or weight <= 0:
            continue
        edges.append(
            _Edge(
                kind="ground",
                from_node=(station, bucket),
                to_node=(station, (bucket + 1) % buckets),
                weight=weight,
                market_id=None,
                direction=None,
                flight_hours=0.0,
            )
        )
    return edges


def _sort_key(edge: _Edge, remaining: int):
    """Deterministic candidate-selection key: prefer the largest-weight edge (fewer,
    cleaner cycles), then break remaining ties by a fixed, arbitrary-but-stable order."""
    return (-remaining, edge.kind, edge.market_id or "", edge.direction or "", edge.to_node)


def _align_cycle(cycle: list[_Edge]) -> list[_Edge]:
    """Rotate a cycle to start at its lexicographically smallest node, purely so
    output is human-readable and independent of which edge the walk happened to
    start from -- the cycle's identity (which edges, which weight) is unaffected."""
    start = min(range(len(cycle)), key=lambda i: cycle[i].from_node)
    return cycle[start:] + cycle[:start]


def _extract_cycles(edges: list[_Edge]) -> list[tuple[list[_Edge], int]]:
    """Decompose a weighted directed multigraph -- assumed to be a circulation (zero
    net divergence at every node) -- into simple directed cycles with weights.

    Generic and domain-agnostic: knows nothing about markets, aircraft, or buckets,
    only about nodes and weighted edges between them. Never mutates `edges`; works
    on an internal copy of the weights.
    """
    n = len(edges)
    remaining = [e.weight for e in edges]
    adjacency: dict[Node, list[int]] = {}
    for i, e in enumerate(edges):
        adjacency.setdefault(e.from_node, []).append(i)

    def pick(candidates: list[int]) -> int | None:
        live = [i for i in candidates if remaining[i] > 0]
        if not live:
            return None
        return min(live, key=lambda i: _sort_key(edges[i], remaining[i]))

    results: list[tuple[list[_Edge], int]] = []
    all_indices = list(range(n))

    while True:
        start_idx = pick(all_indices)
        if start_idx is None:
            break

        path: list[int] = [start_idx]
        first_seen: dict[Node, int] = {edges[start_idx].from_node: 0}
        current = edges[start_idx].to_node

        while current not in first_seen:
            first_seen[current] = len(path)
            next_idx = pick(adjacency.get(current, []))
            if next_idx is None:
                raise RuntimeError(
                    f"tail_routing: no outflow with remaining weight at node {current} -- "
                    "the input graph is not a true circulation (conservation broken)"
                )
            path.append(next_idx)
            current = edges[next_idx].to_node

        cycle_start = first_seen[current]
        cycle_indices = path[cycle_start:]
        bottleneck = min(remaining[i] for i in cycle_indices)
        for i in cycle_indices:
            remaining[i] -= bottleneck

        cycle_edges = _align_cycle([edges[i] for i in cycle_indices])
        results.append((cycle_edges, bottleneck))

    return results


def decompose_flow_into_rotations(
    scheduled_legs: list[ScheduledLegGroup],
    ground_occupancy: dict[tuple[str, str, int], int],
    markets: list[Market],
    economics: dict[tuple[str, str], LegEconomics],
    timespace_types: list[str],
    buckets: int,
) -> list[TailRotation]:
    """Decompose every type's solved circulation into `TailRotation`s.

    Pure function -- no CP-SAT, no I/O. `scheduled_legs`/`ground_occupancy` are
    exactly what `anos.optimize.timespace_assignment._extract_timespace_plan`
    extracts from a solved model; `markets`/`economics` resolve station identities
    and per-leg flight hours (see `_edges_for_type`).
    """
    markets_by_id = {m.market_id: m for m in markets}
    rotations: list[TailRotation] = []

    for type_code in sorted(set(timespace_types)):
        edges = _edges_for_type(
            type_code, scheduled_legs, ground_occupancy, markets_by_id, economics, buckets
        )
        if not edges:
            continue

        for cycle_edges, weight in _extract_cycles(edges):
            legs = tuple(
                TailLegAssignment(
                    kind=e.kind,
                    from_station=e.from_node[0],
                    to_station=e.to_node[0],
                    from_bucket=e.from_node[1],
                    to_bucket=e.to_node[1],
                    market_id=e.market_id,
                    direction=e.direction,
                    flight_hours=e.flight_hours,
                )
                for e in cycle_edges
            )
            rotations.append(
                TailRotation(
                    type_code=type_code,
                    tail_count=weight,
                    legs=legs,
                    daily_flight_hours=sum(e.flight_hours for e in cycle_edges),
                )
            )

    return rotations
