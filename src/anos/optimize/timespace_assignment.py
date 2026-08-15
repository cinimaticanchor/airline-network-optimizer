"""Time-space network fleet assignment -- a trimmed first increment.

Everywhere else in this package, "fleet capacity" means one aggregate daily
aircraft-hours budget per type (see `anos.optimize.fleet_assignment`). That is an
approximation: it never checks that a specific hour of the day's worth of capacity
actually exists, only that the day's *total* hours add up. This module replaces that
approximation, for a chosen subset of aircraft types, with the real thing -- a
multicommodity time-space network flow model (Hane et al. 1995): nodes are
(station, time-bucket) pairs, flight arcs move a unit of an aircraft type's count
from a departure node to an "available again" node (block time plus planned turn, not
just landing), ground arcs let a unit sit, and flow must balance at every node.

**Why enumerated buckets, not a joint schedule-design/Benders formulation.** This
package has no input timetable -- `solve_network()` only ever chooses a daily
*count*, never a time. Real FAM implementations are normally handed an
already-timed schedule; giving this model one means it necessarily also decides a
piece of "schedule design" (which bucket each unit of frequency departs in), not
just aircraft-type assignment. A fully joint formulation (Benders decomposition, as
the itinerary-based-FAM literature does) would be a second solver paradigm with no
precedent anywhere in this codebase, decided in the same phase that also has to get
the time-space graph itself right -- two large, coupled sources of risk. Enumerating
a modest number of candidate buckets and letting CP-SAT pick one per unit of
frequency keeps the "enumerate the small integer choice exactly as booleans/arcs"
philosophy every other extension in this package already uses (see
`fleet_assignment`'s own module docstring), at the cost of only approximating
real time-of-day (a market's departures all effectively share one representative
time within whichever bucket they're assigned).

**Sibling function, not a flag on `solve_network()`.** A bug or scale blowup in the
flow-network construction below then cannot regress the existing, already
non-regression-tested aggregate solver -- it structurally isn't in that call path.
Connections, rotations and fare elasticity are not accepted here in v1: mixing this
module's precise flow-balance clock with `anos.network.dayparts`' coarse heuristic
clock in the same phase that also has to validate the flow network itself would be
compounding, not additive, risk. Integrating them is future work once this core is
validated standalone.

**Non-goals, stated explicitly so they are not mistaken for oversights.** Bucket
choice affects only fleet-*capacity* feasibility -- demand, fares and revenue are
untouched, exactly as `fleet_assignment.demand_curve`/`fare_elasticity` compute them.
There is no time-of-day demand or fare sensitivity (red-eye vs. peak pricing) in this
increment. Airport slot caps stay one number per day (the reference data has no
per-bucket slot capacity to enforce against). Aircraft are aggregate counts, not
individual tails -- see `anos.models.ScheduledLegGroup`'s docstring.

**`NetworkPlan.ground_occupancy` can list a station a type never actually flies to.**
A station with zero scheduled frequency for a type has no flight arcs touching its
`ground[t,s,b]` variables, so they are only constrained by a trivial self-loop
equality (`ground[b-1] == ground[b]`, i.e. "whatever sits here stays sitting") --
CP-SAT is free to leave such an isolated variable at any value in its domain, since
it carries no cost and touches no other constraint. This is harmless for the
optimisation (frequencies, revenue and feasibility are entirely unaffected -- these
variables influence nothing else in the model) but means a station can appear in
`ground_occupancy` with a nonzero count while never appearing in `scheduled_legs`,
which `anos.optimize.tail_routing` will then report as a small "phantom idle
rotation" parked at that station forever. Confirmed empirically during development,
not merely theoretical -- see `tests/test_timespace_assignment.py`'s
`test_ground_occupancy_is_populated_unconditionally`, which asserts the real,
weaker guarantee (every such station is economically *eligible* for the type, per
`build_economics()`) rather than the false stronger one (every such station was
actually *scheduled*).

**The flow network enforces fleet count, not per-airframe daily utilisation** --
those are different constraints. Flow conservation only guarantees the total number
of airframes "in the network" (airborne or committed) never exceeds what's owned; it
has no notion of a single airframe's own daily-hours ceiling (crew duty, maintenance
windows), which is what `params` via `daily_aircraft_hours()` approximates. This
module therefore keeps the *aggregate hours* cap active for every type, including
ones scoped into the flow network -- making the time-space model a strict refinement
of the aggregate one (more constraints; its optimum can only be <= the aggregate
model's) rather than a different, sometimes-looser one. Dropping the hours cap for
timespace-scoped types was tried during development and let the solver pack far more
than `max_daily_util_h`/day into tightly-banked buckets -- caught immediately by
`anos.optimize.tail_feasibility.check_plan()`'s existing utilisation check, which is
exactly why keeping both constraints is the right call, not a workaround.

**Scope trim for this increment.** Validate on 2-3 large single-fleet types (e.g.
`["A320neo", "A321neo"]`) before ever widening -- back-of-envelope sizing across the
full 14-type/59-airport network is on the order of 10,000+ new variables against a
model that currently solves in single digits of seconds; this increment's whole
purpose is to prove the mechanism correct at a scale cheap enough to validate first.
"""

from __future__ import annotations

import math
import time
from datetime import date

from ortools.sat.python import cp_model

from anos.config import Params, load_params
from anos.costs.economics import build_economics
from anos.data.fleet_timeline import daily_aircraft_hours, fleet_on
from anos.data.loaders import (
    load_aircraft_types,
    load_airports,
    load_markets,
    load_timespace_config,
)
from anos.forecast.demand import demand_curve, effective_demand
from anos.models import (
    FleetSnapshot,
    LegEconomics,
    Market,
    MarketPlan,
    NetworkPlan,
    ScheduledLegGroup,
    TailRotation,
)
from anos.network.timespace_clock import avail_bucket, bucket_of, n_buckets, straddles_anchor
from anos.optimize.fleet_assignment import InfeasibleNetworkError, _market_by_id, _preflight
from anos.optimize.tail_routing import decompose_flow_into_rotations

HOURS_SCALE = 100  # aircraft-hours -> centihours, matching fleet_assignment.py


def solve_network_timespace(
    target: date,
    *,
    timespace_types: list[str],
    month: int | None = None,
    markets: list[Market] | None = None,
    fleet: FleetSnapshot | None = None,
    params: Params | None = None,
    probabilistic_fleet: bool = False,
    enforce_min_service: bool = True,
    scenario: str = "baseline",
    verbose: bool = False,
    fast_turn: bool = False,
    n_buckets_override: int | None = None,
    build_tail_rotations: bool = False,
) -> NetworkPlan:
    """Solve fleet assignment with real time-space flow-balance for `timespace_types`.

    Types not in `timespace_types` keep using the aggregate daily-hours budget from
    `anos.optimize.fleet_assignment`, unmodified, in the same model -- both mechanisms
    coexist correctly because they constrain disjoint sets of `freq` variables.

    Args:
        timespace_types: aircraft type codes to capacity-check via the time-space
            network. Required, non-empty -- there is no sensible inert default for a
            function whose entire purpose is time-space capacity checking.
        n_buckets_override: override `data/timespace.yaml`'s bucket count.
        build_tail_rotations: decompose the solved circulation into concrete example
            rotations (`anos.optimize.tail_routing`), populating
            `NetworkPlan.tail_rotations`. Default off -- pure post-processing on an
            already-solved plan, so it costs nothing when unused.
        (all other args match `anos.optimize.fleet_assignment.solve_network`.)

    Raises:
        ValueError: if `timespace_types` is empty.
        InfeasibleNetworkError: no feasible network, or a (market, type) pair whose
            one-way block+turn time spans a full cyclic day at the configured bucket
            width (increase buckets or exclude that type).
    """
    if not timespace_types:
        raise ValueError("timespace_types must be a non-empty list of aircraft type codes")

    par = params or load_params()
    mkts = markets if markets is not None else load_markets()
    snapshot = fleet or fleet_on(target, probabilistic=probabilistic_fleet, params=par)
    season_month = month if month is not None else target.month

    types = load_aircraft_types()
    ports = load_airports()
    economics = build_economics(mkts, params=par, fast_turn=fast_turn)
    hours_available = daily_aircraft_hours(snapshot)

    problems = _preflight(mkts, economics)
    if problems:
        raise InfeasibleNetworkError(
            "network cannot be flown as specified:\n  - " + "\n  - ".join(problems)
        )

    ts_config = load_timespace_config()
    buckets = n_buckets_override or n_buckets(ts_config)
    anchor = bucket_of(float(ts_config["anchor_hour"]), ts_config)
    ts_types = set(timespace_types)
    fleet_count = {t: math.floor(snapshot.available.get(t, 0.0)) for t in ts_types}

    # -- precompute each eligible (market, timespace-type)'s dep->avail bucket map --
    dep_to_avail: dict[tuple[str, str], dict[int, int]] = {}
    for market in mkts:
        for t in ts_types:
            key = (market.market_id, t)
            if key not in economics or fleet_count.get(t, 0) <= 0:
                continue
            duration_h = economics[key].aircraft_hours_consumed / 2.0  # one-way block + turn
            mapping: dict[int, int] = {}
            for b in range(buckets):
                avail_b, duration_buckets = avail_bucket(b, duration_h, ts_config)
                if duration_buckets >= buckets:
                    raise InfeasibleNetworkError(
                        f"{market.market_id}/{t}: one-way block+turn ({duration_h:.1f}h) "
                        f"spans a full cyclic day at {buckets} buckets -- increase buckets "
                        "or exclude this type from timespace_types"
                    )
                mapping[b] = avail_b
            dep_to_avail[key] = mapping

    stations_for_type: dict[str, set[str]] = {t: set() for t in ts_types}
    for mid, t in dep_to_avail:
        m = _market_by_id(mkts, mid)
        stations_for_type[t].add(m.origin)
        stations_for_type[t].add(m.destination)

    model = cp_model.CpModel()

    freq: dict[tuple[str, str], cp_model.IntVar] = {}
    out_arc: dict[tuple[str, str, int], cp_model.IntVar] = {}
    ret_arc: dict[tuple[str, str, int], cp_model.IntVar] = {}
    ground: dict[tuple[str, str, int], cp_model.IntVar] = {}
    level: dict[tuple[str, int], cp_model.IntVar] = {}
    pax: dict[str, cp_model.IntVar] = {}
    total_freq: dict[str, cp_model.IntVar] = {}
    objective_terms: list[cp_model.LinearExpr] = []

    # -- ground arcs -----------------------------------------------------------
    for t in ts_types:
        if fleet_count.get(t, 0) <= 0:
            continue
        for s in stations_for_type[t]:
            for b in range(buckets):
                ground[(t, s, b)] = model.NewIntVar(0, fleet_count[t], f"gr[{t},{s},{b}]")

    # -- flight arcs (out/return) and freq linkage for timespace-scoped types ---
    arrivals: dict[tuple[str, str, int], list[cp_model.IntVar]] = {}
    departures: dict[tuple[str, str, int], list[cp_model.IntVar]] = {}
    arc_records: list[tuple[cp_model.IntVar, str, int, int]] = []  # (var, type, dep_bucket, avail_bucket)

    for market in mkts:
        mid = market.market_id
        for t in ts_types:
            key = (mid, t)
            if key not in dep_to_avail:
                continue
            mapping = dep_to_avail[key]
            outs, rets = [], []
            for b in range(buckets):
                avail_b = mapping[b]

                ov = model.NewIntVar(0, market.max_daily_freq, f"out[{mid},{t},{b}]")
                out_arc[(mid, t, b)] = ov
                departures.setdefault((t, market.origin, b), []).append(ov)
                arrivals.setdefault((t, market.destination, avail_b), []).append(ov)
                arc_records.append((ov, t, b, avail_b))
                outs.append(ov)

                rv = model.NewIntVar(0, market.max_daily_freq, f"ret[{mid},{t},{b}]")
                ret_arc[(mid, t, b)] = rv
                departures.setdefault((t, market.destination, b), []).append(rv)
                arrivals.setdefault((t, market.origin, avail_b), []).append(rv)
                arc_records.append((rv, t, b, avail_b))
                rets.append(rv)

            freq[(mid, t)] = model.NewIntVar(0, market.max_daily_freq, f"f[{mid},{t}]")
            model.Add(freq[(mid, t)] == sum(outs))
            model.Add(freq[(mid, t)] == sum(rets))

    # -- non-timespace-scoped types (and timespace-scoped types with zero current
    # fleet count) keep plain frequency variables here, unconstrained by arcs --
    # the aggregate fleet-capacity block below covers them, exactly as in
    # fleet_assignment.solve_network. A timespace-scoped type with fleet_count<=0
    # never gets a `dep_to_avail` entry (see above), so it's excluded from the
    # flight-arc loop entirely -- without this branch it would have no `freq`
    # variable created at all, and the demand loop below would KeyError the moment
    # it's economically eligible for a market (eligibility is a range/widebody
    # check, unrelated to whether any of the type has actually been delivered yet).
    for market in mkts:
        mid = market.market_id
        for t in sorted(t for (m, t) in economics if m == mid):
            if t in ts_types and fleet_count.get(t, 0) > 0:
                continue  # already created in the flight-arc loop above
            if (mid, t) in freq:
                continue
            freq[(mid, t)] = model.NewIntVar(0, market.max_daily_freq, f"f[{mid},{t}]")

    # -- flow balance: aircraft in == aircraft out, every (type, station, bucket) --
    for t in ts_types:
        if fleet_count.get(t, 0) <= 0:
            continue
        for s in stations_for_type[t]:
            for b in range(buckets):
                prev_b = (b - 1) % buckets
                inflow = [ground[(t, s, prev_b)], *arrivals.get((t, s, b), [])]
                outflow = [ground[(t, s, b)], *departures.get((t, s, b), [])]
                model.Add(sum(inflow) == sum(outflow))

    # -- fleet-count cap: tokens crossing the anchor cross-section <= fleet count --
    for t in ts_types:
        if fleet_count.get(t, 0) <= 0:
            for market in mkts:
                key = (market.market_id, t)
                if key in freq:
                    model.Add(freq[key] == 0)
            continue
        ground_terms = [ground[(t, s, anchor)] for s in stations_for_type[t]]
        straddle_terms = [
            var
            for (var, t2, b, avail_b) in arc_records
            if t2 == t and straddles_anchor(b, avail_b, anchor, buckets)
        ]
        model.Add(sum(ground_terms) + sum(straddle_terms) <= fleet_count[t])

    # -- per-market demand, pax and objective -- unchanged from fleet_assignment.py,
    # duplicated rather than shared (see module docstring on why). ----------------
    for market in mkts:
        mid = market.market_id
        eligible_types = sorted(t for (m, t) in economics if m == mid)
        if not eligible_types:
            continue

        curve = demand_curve(market, season_month, par)
        max_freq = market.max_daily_freq
        min_freq = market.min_daily_freq if enforce_min_service else 0

        total_freq[mid] = model.NewIntVar(min_freq, max_freq, f"F[{mid}]")
        model.Add(total_freq[mid] == sum(freq[(mid, t)] for t in eligible_types))

        levels = range(min_freq, max_freq + 1)
        for k in levels:
            level[(mid, k)] = model.NewBoolVar(f"y[{mid},{k}]")
        model.AddExactlyOne(level[(mid, k)] for k in levels)
        model.Add(total_freq[mid] == sum(k * level[(mid, k)] for k in levels))

        demand_cap = max(int(round(curve.get(k, 0.0))) for k in levels)
        pax[mid] = model.NewIntVar(0, demand_cap, f"pax[{mid}]")
        model.Add(
            pax[mid] <= sum(int(round(curve.get(k, 0.0))) * level[(mid, k)] for k in levels)
        )
        model.Add(
            pax[mid]
            <= sum(economics[(mid, t)].seats_round_trip * freq[(mid, t)] for t in eligible_types)
        )

        is_domestic = ports[market.origin].is_domestic and ports[market.destination].is_domestic
        margin_per_pax = market.avg_fare_usd - par.cost_per_pax(domestic=is_domestic)
        objective_terms.append(int(round(2 * margin_per_pax)) * pax[mid])

        for t in eligible_types:
            trip_cost = int(round(economics[(mid, t)].cost_per_round_trip_usd))
            objective_terms.append(-trip_cost * freq[(mid, t)])

    # -- aggregate fleet capacity ---------------------------------------------
    # Applied to EVERY type, including timespace-scoped ones -- not a redundant
    # check. The flow network only enforces fleet *count* (never more airframes in
    # the network than owned); it has no notion of a per-airframe daily-utilisation
    # ceiling (crew duty, maintenance windows), which is exactly what this aggregate
    # hours cap represents. Without it, the flow network alone can pack far more than
    # `max_daily_util_h` per airframe into tightly-banked buckets -- confirmed during
    # development, where `tail_feasibility.check_plan()` correctly flagged such a
    # plan as a utilisation blocker even though flow conservation held. Keeping this
    # constraint makes the time-space model a strict refinement of the aggregate one
    # for these types (more constraints, so its optimum can only be <= the aggregate
    # model's), not a different, sometimes-looser one.
    for code in types:
        hours = hours_available.get(code, 0.0)
        consuming = [(mid, t) for (mid, t) in freq if t == code]
        if not consuming:
            continue
        if hours <= 0:
            for key in consuming:
                model.Add(freq[key] == 0)
            continue
        model.Add(
            sum(
                int(round(economics[key].aircraft_hours_consumed * HOURS_SCALE)) * freq[key]
                for key in consuming
            )
            <= int(round(hours * HOURS_SCALE))
        )

    # -- airport slot allocations (one number per day, same as fleet_assignment) --
    for iata, airport in ports.items():
        touching = [
            (mid, t)
            for (mid, t) in freq
            if iata in (_market_by_id(mkts, mid).origin, _market_by_id(mkts, mid).destination)
        ]
        if not touching:
            continue
        model.Add(sum(freq[key] for key in touching) <= airport.daily_slot_cap)

    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = par.solver_max_seconds
    solver.parameters.num_workers = par.solver_num_workers
    solver.parameters.relative_gap_limit = par.solver_relative_gap
    solver.parameters.log_search_progress = verbose

    started = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - started

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise InfeasibleNetworkError(
            f"solver returned {solver.StatusName(status)} -- no feasible network exists "
            "for this fleet and this set of service floors"
        )

    return _extract_timespace_plan(
        solver=solver,
        status=solver.StatusName(status),
        elapsed=elapsed,
        target=target,
        month=season_month,
        markets=mkts,
        economics=economics,
        freq=freq,
        pax=pax,
        hours_available=hours_available,
        params=par,
        scenario=scenario,
        out_arc=out_arc,
        ret_arc=ret_arc,
        ground=ground,
        dep_to_avail=dep_to_avail,
        timespace_types=sorted(ts_types),
        buckets=buckets,
        build_tail_rotations=build_tail_rotations,
    )


def _extract_timespace_plan(
    *,
    solver: cp_model.CpSolver,
    status: str,
    elapsed: float,
    target: date,
    month: int,
    markets: list[Market],
    economics: dict[tuple[str, str], LegEconomics],
    freq: dict[tuple[str, str], cp_model.IntVar],
    pax: dict[str, cp_model.IntVar],
    hours_available: dict[str, float],
    params: Params,
    scenario: str,
    out_arc: dict[tuple[str, str, int], cp_model.IntVar],
    ret_arc: dict[tuple[str, str, int], cp_model.IntVar],
    ground: dict[tuple[str, str, int], cp_model.IntVar],
    dep_to_avail: dict[tuple[str, str], dict[int, int]],
    timespace_types: list[str],
    buckets: int,
    build_tail_rotations: bool,
) -> NetworkPlan:
    """Turn solver variable values back into domain objects.

    `fleet_used`/`fleet_available` are populated exactly as `fleet_assignment.py`
    does -- post-hoc from solved `freq` values -- even though capacity was *enforced*
    via the flow network for `timespace_types`. This is deliberate: it means every
    existing `NetworkPlan` consumer (`anos.optimize.tail_feasibility.check_plan`,
    `anos.report.html_report`, every `cli.py` printer) works unmodified on a plan
    from this function.
    """
    ports = load_airports()
    market_plans: list[MarketPlan] = []
    fleet_used: dict[str, float] = {}

    for market in markets:
        mid = market.market_id
        if mid not in pax:
            continue

        frequencies = {
            t: solver.Value(var)
            for (m, t), var in freq.items()
            if m == mid and solver.Value(var) > 0
        }
        total = sum(frequencies.values())
        carried = float(solver.Value(pax[mid]))

        seats = sum(economics[(mid, t)].seats_round_trip * n for t, n in frequencies.items())
        trip_cost = sum(
            economics[(mid, t)].cost_per_round_trip_usd * n for t, n in frequencies.items()
        )

        for t, n in frequencies.items():
            fleet_used[t] = (
                fleet_used.get(t, 0.0) + economics[(mid, t)].aircraft_hours_consumed * n
            )

        is_domestic = ports[market.origin].is_domestic and ports[market.destination].is_domestic
        revenue = 2 * carried * market.avg_fare_usd
        cost = trip_cost + 2 * carried * params.cost_per_pax(domestic=is_domestic)

        market_plans.append(
            MarketPlan(
                market_id=mid,
                frequencies=frequencies,
                total_frequency=total,
                seats_offered=seats,
                pax_carried=carried,
                demand_effective=effective_demand(market, total, month, params),
                revenue_usd=revenue,
                cost_usd=cost,
            )
        )

    scheduled_legs: list[ScheduledLegGroup] = []
    for (mid, t, b), var in out_arc.items():
        n = solver.Value(var)
        if n > 0:
            scheduled_legs.append(
                ScheduledLegGroup(
                    market_id=mid, type_code=t, direction="outbound",
                    departure_bucket=b, avail_bucket=dep_to_avail[(mid, t)][b], count=n,
                )
            )
    for (mid, t, b), var in ret_arc.items():
        n = solver.Value(var)
        if n > 0:
            scheduled_legs.append(
                ScheduledLegGroup(
                    market_id=mid, type_code=t, direction="return",
                    departure_bucket=b, avail_bucket=dep_to_avail[(mid, t)][b], count=n,
                )
            )

    ground_occupancy: dict[tuple[str, str, int], int] = {
        key: solver.Value(var) for key, var in ground.items() if solver.Value(var) > 0
    }

    tail_rotations: list[TailRotation] = []
    if build_tail_rotations:
        tail_rotations = decompose_flow_into_rotations(
            scheduled_legs=scheduled_legs,
            ground_occupancy=ground_occupancy,
            markets=markets,
            economics=economics,
            timespace_types=timespace_types,
            buckets=buckets,
        )

    return NetworkPlan(
        as_of=target,
        month=month,
        market_plans=market_plans,
        fleet_used=fleet_used,
        fleet_available=dict(hours_available),
        solver_status=status,
        objective_usd=solver.ObjectiveValue(),
        solve_seconds=elapsed,
        scenario=scenario,
        scheduled_legs=scheduled_legs,
        timespace_types=timespace_types,
        ground_occupancy=ground_occupancy,
        tail_rotations=tail_rotations,
    )
