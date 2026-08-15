"""The optimisation core: which aircraft type flies which market, how often.

Formulated as a mixed-integer program over one representative day in a given month,
solved with OR-Tools CP-SAT.

    maximise   sum over markets of  ( fare x passengers carried
                                      - passenger-driven cost
                                      - cost of the trips flown )

    subject to
      * fleet capacity -- aircraft-hours consumed by type cannot exceed the hours the
        fleet timeline says are actually flyable that month
      * range and airport compatibility (enforced by omitting ineligible pairings)
      * passengers carried cannot exceed demand, and cannot exceed seats offered
      * demand itself rises concavely with frequency (schedule quality)
      * minimum service floors on strategic and regulated markets
      * per-airport slot allocations

**Why this stays linear.** Demand depends on frequency non-linearly, which would
normally make the objective non-convex. Because daily frequency is a small integer,
the model instead enumerates every feasible frequency level per market as a one-hot
set of booleans, and reads the demand for that level straight off the curve. The
relationship is represented exactly, with no linearisation error, and the model stays
a MIP that CP-SAT solves in seconds.

**Units.** CP-SAT works in integers. Money is whole US dollars; aircraft-hours are
scaled to centihours. Passengers are whole people. All rounding is applied once, at
model build time.

**Opt-in extensions.** Keyword-only flags -- `fast_turn`, `enable_fare_elasticity`,
`enable_connections` (with `hubs`), `enable_rotations` (with `hubs` and
`rotation_squad`), `carbon_priced`, `enable_recapture` (requires `enable_connections`),
and `enable_interline` -- each default to their inert value, in which case the model
built here is identical to the one described above. See the module docstrings in
`anos.network.dayparts`, `anos.network.itinerary` and `anos.forecast.fare_elasticity`
for what each extension represents and why it is scoped the way it is.
"""

from __future__ import annotations

import time
from datetime import date

from ortools.sat.python import cp_model

from anos.config import Params, load_params
from anos.costs.economics import build_economics, build_rotation_economics
from anos.data.fleet_timeline import daily_aircraft_hours, fleet_on
from anos.data.loaders import (
    load_aircraft_types,
    load_airports,
    load_hub_topology,
    load_markets,
)
from anos.forecast.demand import demand_curve, seasonal_demand
from anos.forecast.fare_elasticity import demand_multiplier_for_fare, fare_levels
from anos.models import (
    ConnectionCandidate,
    FleetSnapshot,
    LegEconomics,
    Market,
    MarketPlan,
    NetworkPlan,
    RotationCandidate,
    RotationEconomics,
    great_circle_km,
)
from anos.network.dayparts import connection_feasible, daypart_buckets, feeder_arrival_daypart
from anos.network.itinerary import build_connection_candidates, build_rotation_candidates

HOURS_SCALE = 100  # aircraft-hours -> centihours


class InfeasibleNetworkError(RuntimeError):
    """Raised when the network as specified cannot be flown at all."""


def _preflight(
    markets: list[Market],
    economics: dict[tuple[str, str], LegEconomics],
) -> list[str]:
    """Catch structurally impossible requirements before handing work to the solver.

    A market with a mandatory minimum frequency and no aircraft type able to fly it
    makes the whole model infeasible, and CP-SAT will report that as a bare
    INFEASIBLE with no indication of which market caused it.
    """
    problems: list[str] = []
    for market in markets:
        eligible = [t for (m, t) in economics if m == market.market_id]
        if not eligible and market.min_daily_freq > 0:
            problems.append(
                f"{market.market_id}: minimum service of {market.min_daily_freq}/day "
                "required but no aircraft type can fly the sector"
            )
    return problems


def solve_network(
    target: date,
    *,
    month: int | None = None,
    markets: list[Market] | None = None,
    fleet: FleetSnapshot | None = None,
    params: Params | None = None,
    probabilistic_fleet: bool = False,
    enforce_min_service: bool = True,
    scenario: str = "baseline",
    verbose: bool = False,
    fast_turn: bool = False,
    enable_fare_elasticity: bool = False,
    enable_connections: bool = False,
    hubs: list[str] | None = None,
    enable_rotations: bool = False,
    rotation_squad: list[str] | None = None,
    carbon_priced: bool = False,
    enable_recapture: bool = False,
    enable_interline: bool = False,
) -> NetworkPlan:
    """Solve the fleet-assignment problem for one representative day.

    Args:
        target: the date whose fleet availability the plan is built against.
        month: month (1-12) whose seasonality applies; defaults to the target's month.
        markets: market set override.
        fleet: fleet snapshot override; otherwise derived from the fleet timeline.
        params: economic parameter override.
        probabilistic_fleet: risk-adjust delivery dates when deriving the fleet.
        enforce_min_service: honour per-market minimum service floors.
        scenario: label recorded on the returned plan.
        verbose: stream solver progress.
        fast_turn: use `params.fast_turn_buffer_factor` instead of `turn_buffer_factor`
            (a minimum-connect-time programme), freeing aircraft-hours for more
            departures at the same service level. Default reproduces today's economics.
        enable_fare_elasticity: let the solver choose a fare level per market from
            `params.fare_multipliers`, trading fare for demand via a stated elasticity
            assumption (see `anos.forecast.fare_elasticity`). Default keeps every
            market at its reference fare -- byte-identical to today's objective.
        enable_connections: allow one-stop itineraries through `hubs` to compete for
            feeder/trunk spare seats (banking). Requires `hubs`; a no-op without it.
        hubs: airports connections and/or rotations may route through. The same list
            serves both -- a secondary regional hub is a parameter, not new code.
        enable_rotations: allow `rotation_squad` markets sharing a hub to be served by
            a shared multi-stop loop instead of two separate round trips. Requires
            `hubs` and `rotation_squad`; a no-op without both.
        rotation_squad: market_ids eligible to be pooled into rotation loops --
            typically `anos.network.itinerary.low_load_factor_squad(baseline_plan)`.
        carbon_priced: fold a blended EU/UK ETS-style carbon cost into round trips
            touching `params.carbon_priced_countries`. Default reproduces today's costs.
        enable_recapture: let a market's own connection candidate absorb a share of
            that market's nonstop spill (`params.connection_recapture_rate`), instead
            of assuming spill is lost entirely. Requires `enable_connections`; a no-op
            without it.
        enable_interline: let markets flagged `interline_available` in markets.csv
            (structurally unbankable at any hub we control) sell a partner/codeshare
            itinerary at a stated flat prorate (`params.interline_prorate_usd_per_pax`).

    Raises:
        InfeasibleNetworkError: if no feasible network exists.
    """
    par = params or load_params()
    mkts = markets if markets is not None else load_markets()
    snapshot = fleet or fleet_on(target, probabilistic=probabilistic_fleet, params=par)
    season_month = month if month is not None else target.month

    types = load_aircraft_types()
    ports = load_airports()
    economics = build_economics(mkts, params=par, fast_turn=fast_turn, carbon_priced=carbon_priced)
    hours_available = daily_aircraft_hours(snapshot)

    problems = _preflight(mkts, economics)
    if problems:
        raise InfeasibleNetworkError(
            "network cannot be flown as specified:\n  - " + "\n  - ".join(problems)
        )

    connections: list[ConnectionCandidate] = (
        build_connection_candidates(mkts, hubs) if (enable_connections and hubs) else []
    )
    rotations: list[RotationCandidate] = (
        build_rotation_candidates(mkts, rotation_squad, hubs, max_stops=2)
        if (enable_rotations and hubs and rotation_squad)
        else []
    )
    rot_econ: dict[tuple[str, str], RotationEconomics] = (
        build_rotation_economics(rotations, params=par) if rotations else {}
    )

    feeder_candidates: dict[str, list[ConnectionCandidate]] = {}
    trunk_candidates: dict[str, list[ConnectionCandidate]] = {}
    for c in connections:
        feeder_candidates.setdefault(c.feeder_market_id, []).append(c)
        trunk_candidates.setdefault(c.trunk_market_id, []).append(c)

    rotation_by_leg: dict[str, list[RotationCandidate]] = {}
    rotation_ids_by_airport: dict[str, list[str]] = {}
    for r in rotations:
        for leg_mid in r.leg_market_ids:
            rotation_by_leg.setdefault(leg_mid, []).append(r)
        for iata in (r.hub, *r.stops):
            rotation_ids_by_airport.setdefault(iata, []).append(r.rotation_id)

    rot_types_by_rotation: dict[str, list[str]] = {}
    for rid, t in rot_econ:
        rot_types_by_rotation.setdefault(rid, []).append(t)

    model = cp_model.CpModel()

    freq: dict[tuple[str, str], cp_model.IntVar] = {}
    level: dict[tuple[str, int], cp_model.IntVar] = {}
    pax: dict[str, cp_model.IntVar] = {}
    total_freq: dict[str, cp_model.IntVar] = {}
    fare_choice_vars: dict[str, tuple[list[float], dict[int, cp_model.IntVar]]] = {}

    objective_terms: list[cp_model.LinearExpr] = []

    # -- rotation loop frequencies (built before the per-market loop: markets that
    # are rotation legs need to reference these when summing total_freq) ----------
    rot_freq: dict[tuple[str, str], cp_model.IntVar] = {}
    if rotations:
        by_mid = {m.market_id: m for m in mkts}
        for r in rotations:
            leg_a, leg_b = (by_mid[mid] for mid in r.leg_market_ids)
            upper = min(leg_a.max_daily_freq, leg_b.max_daily_freq)
            for t in rot_types_by_rotation.get(r.rotation_id, []):
                rot_freq[(r.rotation_id, t)] = model.NewIntVar(
                    0, upper, f"rf[{r.rotation_id},{t}]"
                )

    # -- connection (banking) variables, built before the per-market loop for the
    # same reason: feeder/trunk seat constraints need to reference them -----------
    topology = load_hub_topology() if connections else None
    n_dayparts = daypart_buckets(topology) if topology else 0
    daypart_markets = sorted(set(feeder_candidates) | set(trunk_candidates))

    bank_daypart: dict[tuple[str, int], cp_model.IntVar] = {}
    for mid in daypart_markets:
        for d in range(n_dayparts):
            bank_daypart[(mid, d)] = model.NewBoolVar(f"bd[{mid},{d}]")
        model.AddExactlyOne(bank_daypart[(mid, d)] for d in range(n_dayparts))

    connect_pax: dict[str, cp_model.IntVar] = {}
    connect_ok: dict[str, cp_model.IntVar] = {}
    connect_cap: dict[str, int] = {}
    for c in connections:
        market = next(m for m in mkts if m.market_id == c.market_id)
        cap = int(round(seasonal_demand(market, season_month, par) * par.connection_capture_rate))
        if cap <= 0:
            continue

        feeder_market = next(m for m in mkts if m.market_id == c.feeder_market_id)
        origin = ports[feeder_market.origin]
        hub_port = ports[c.hub]
        distance_km = great_circle_km(origin.lat, origin.lon, hub_port.lat, hub_port.lon)

        feasible_pairs = {
            (d_f, d_t)
            for d_f in range(n_dayparts)
            for d_t in range(n_dayparts)
            if connection_feasible(
                feeder_arrival_daypart(d_f, distance_km, origin, hub_port, topology=topology),
                d_t,
                topology=topology,
            )
        }

        # Domain leaves headroom for recapture (below, once demand/level vars exist
        # for every market) without changing behaviour when recapture is off: the
        # gate constraint uses `upper`, which equals `cap` unless enable_recapture.
        upper = cap * 2 if enable_recapture else cap
        connect_cap[c.connection_id] = cap
        connect_ok[c.connection_id] = model.NewBoolVar(f"cok[{c.connection_id}]")
        connect_pax[c.connection_id] = model.NewIntVar(0, upper, f"cpax[{c.connection_id}]")
        model.Add(connect_pax[c.connection_id] <= upper * connect_ok[c.connection_id])

        for d_f in range(n_dayparts):
            for d_t in range(n_dayparts):
                if (d_f, d_t) not in feasible_pairs:
                    model.Add(
                        bank_daypart[(c.feeder_market_id, d_f)]
                        + bank_daypart[(c.trunk_market_id, d_t)]
                        + connect_ok[c.connection_id]
                        <= 2
                    )

        is_domestic = ports[market.origin].is_domestic and ports[market.destination].is_domestic
        margin = (
            market.avg_fare_usd
            - par.cost_per_pax(domestic=is_domestic)
            - par.connection_handling_cost_usd_per_pax
        )
        objective_terms.append(int(round(2 * margin)) * connect_pax[c.connection_id])

    # -- interline: partner-sold itineraries on structurally unbankable markets ----
    # Independent of eligible_types -- a market flagged interline_available may have
    # no aircraft that can fly it at all; that is exactly the point.
    interline_pax: dict[str, cp_model.IntVar] = {}
    if enable_interline:
        for market in mkts:
            if not market.interline_available:
                continue
            cap = int(round(seasonal_demand(market, season_month, par) * par.interline_capture_rate))
            if cap <= 0:
                continue
            interline_pax[market.market_id] = model.NewIntVar(
                0, cap, f"ilpax[{market.market_id}]"
            )
            objective_terms.append(
                int(round(2 * par.interline_prorate_usd_per_pax)) * interline_pax[market.market_id]
            )

    # -- per-market frequency, demand and (optionally) fare-level decisions --------
    market_levels: dict[str, range] = {}
    market_curves: dict[str, dict[int, float]] = {}
    for market in mkts:
        mid = market.market_id
        eligible_types = sorted(t for (m, t) in economics if m == mid)

        if not eligible_types:
            # No aircraft can fly it and no minimum forces the issue: the market is
            # simply unserved. Skip rather than build empty variables for it.
            continue

        curve = demand_curve(market, season_month, par)
        max_freq = market.max_daily_freq
        min_freq = market.min_daily_freq if enforce_min_service else 0
        market_curves[mid] = curve

        for t in eligible_types:
            freq[(mid, t)] = model.NewIntVar(0, max_freq, f"f[{mid},{t}]")

        total_freq[mid] = model.NewIntVar(min_freq, max_freq, f"F[{mid}]")
        rot_terms = [
            rot_freq[(r.rotation_id, t)]
            for r in rotation_by_leg.get(mid, [])
            for t in rot_types_by_rotation.get(r.rotation_id, [])
            if (r.rotation_id, t) in rot_freq
        ]
        model.Add(
            total_freq[mid] == sum(freq[(mid, t)] for t in eligible_types) + sum(rot_terms)
        )

        # One-hot over frequency levels, carrying the demand for that level.
        levels = range(min_freq, max_freq + 1)
        market_levels[mid] = levels
        for k in levels:
            level[(mid, k)] = model.NewBoolVar(f"y[{mid},{k}]")
        model.AddExactlyOne(level[(mid, k)] for k in levels)
        model.Add(total_freq[mid] == sum(k * level[(mid, k)] for k in levels))

        is_domestic = ports[market.origin].is_domestic and ports[market.destination].is_domestic
        base_cost_per_pax = par.cost_per_pax(domestic=is_domestic)

        if enable_fare_elasticity:
            fare_opts = fare_levels(market, par)
            fare_choice = {j: model.NewBoolVar(f"fl[{mid},{j}]") for j in range(len(fare_opts))}
            model.AddExactlyOne(fare_choice.values())
            fare_choice_vars[mid] = (fare_opts, fare_choice)

            pax_fare: dict[int, cp_model.IntVar] = {}
            cap_js: list[int] = []
            for j, fare in enumerate(fare_opts):
                mult = demand_multiplier_for_fare(market, fare, par)
                demand_by_level = {k: int(round(curve.get(k, 0.0) * mult)) for k in levels}
                cap_j = max(demand_by_level.values(), default=0)
                cap_js.append(cap_j)

                combo_terms = []
                for k in levels:
                    c = model.NewBoolVar(f"cb[{mid},{k},{j}]")
                    model.Add(c <= level[(mid, k)])
                    model.Add(c <= fare_choice[j])
                    model.Add(c >= level[(mid, k)] + fare_choice[j] - 1)
                    combo_terms.append((demand_by_level[k], c))

                pax_fare[j] = model.NewIntVar(0, cap_j, f"paxf[{mid},{j}]")
                model.Add(pax_fare[j] <= sum(d * c for d, c in combo_terms))

            demand_cap = max(cap_js, default=0)
            pax[mid] = model.NewIntVar(0, demand_cap, f"pax[{mid}]")
            model.Add(pax[mid] == sum(pax_fare.values()))

            for j, fare in enumerate(fare_opts):
                margin_j = fare - base_cost_per_pax
                objective_terms.append(int(round(2 * margin_j)) * pax_fare[j])
        else:
            demand_cap = max(int(round(curve.get(k, 0.0))) for k in levels)
            pax[mid] = model.NewIntVar(0, demand_cap, f"pax[{mid}]")
            # Passengers cannot exceed the demand the chosen frequency generates...
            model.Add(
                pax[mid]
                <= sum(int(round(curve.get(k, 0.0))) * level[(mid, k)] for k in levels)
            )
            margin_per_pax = market.avg_fare_usd - base_cost_per_pax
            # Both directions of the round trip earn.
            objective_terms.append(int(round(2 * margin_per_pax)) * pax[mid])

        # ...nor the seats actually offered in one direction -- shared with any
        # connecting itineraries using this leg as a feeder or trunk, and topped up
        # by any rotation loops that also serve this market.
        extra_pax_terms = [
            connect_pax[c.connection_id]
            for c in feeder_candidates.get(mid, []) + trunk_candidates.get(mid, [])
            if c.connection_id in connect_pax
        ]
        rot_seat_terms = [
            types[t].seats * rot_freq[(r.rotation_id, t)]
            for r in rotation_by_leg.get(mid, [])
            for t in rot_types_by_rotation.get(r.rotation_id, [])
            if (r.rotation_id, t) in rot_freq
        ]
        model.Add(
            pax[mid] + sum(extra_pax_terms)
            <= sum(
                economics[(mid, t)].seats_round_trip * freq[(mid, t)] for t in eligible_types
            )
            + sum(rot_seat_terms)
        )

        for t in eligible_types:
            trip_cost = int(round(economics[(mid, t)].cost_per_round_trip_usd))
            objective_terms.append(-trip_cost * freq[(mid, t)])

    for key, var in rot_freq.items():
        objective_terms.append(-int(round(rot_econ[key].cost_usd)) * var)

    # -- demand recapture: spilled nonstop demand flows onto the market's own
    # connection candidate, instead of being assumed lost entirely. Needs pax/level
    # vars for every market, which is why this runs after the per-market loop. -----
    if enable_recapture:
        recapture_scale = int(round(par.connection_recapture_rate * 100))
        for c in connections:
            if c.connection_id not in connect_pax or c.market_id not in pax:
                continue
            cap = connect_cap[c.connection_id]
            levels_m = market_levels[c.market_id]
            curve_m = market_curves[c.market_id]
            spill_expr = (
                sum(int(round(curve_m.get(k, 0.0))) * level[(c.market_id, k)] for k in levels_m)
                - pax[c.market_id]
            )
            recapture_actual = model.NewIntVar(0, cap, f"recap[{c.connection_id}]")
            model.Add(100 * recapture_actual <= recapture_scale * spill_expr)
            model.Add(connect_pax[c.connection_id] <= cap + recapture_actual)

    # -- fleet capacity ------------------------------------------------------
    for code in types:
        hours = hours_available.get(code, 0.0)
        consuming = [(mid, t) for (mid, t) in freq if t == code]
        rot_consuming = [(rid, t) for (rid, t) in rot_freq if t == code]
        if not consuming and not rot_consuming:
            continue
        if hours <= 0:
            for key in consuming:
                model.Add(freq[key] == 0)
            for key in rot_consuming:
                model.Add(rot_freq[key] == 0)
            continue
        model.Add(
            sum(
                int(round(economics[key].aircraft_hours_consumed * HOURS_SCALE)) * freq[key]
                for key in consuming
            )
            + sum(
                int(round(rot_econ[key].aircraft_hours_consumed * HOURS_SCALE))
                * rot_freq[key]
                for key in rot_consuming
            )
            <= int(round(hours * HOURS_SCALE))
        )

    # -- airport slot allocations -------------------------------------------
    # Each daily round trip consumes one departure slot at each end; each rotation
    # loop consumes one slot at the hub and one at each stop, not two per market.
    for iata, airport in ports.items():
        touching = [
            (mid, t)
            for (mid, t) in freq
            if iata in (_market_by_id(mkts, mid).origin, _market_by_id(mkts, mid).destination)
        ]
        rot_touching = [
            (rid, t)
            for rid in rotation_ids_by_airport.get(iata, [])
            for t in rot_types_by_rotation.get(rid, [])
            if (rid, t) in rot_freq
        ]
        if not touching and not rot_touching:
            continue
        model.Add(
            sum(freq[key] for key in touching) + sum(rot_freq[key] for key in rot_touching)
            <= airport.daily_slot_cap
        )

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

    return _extract_plan(
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
        connections=connections,
        connect_pax=connect_pax,
        rotations=rotations,
        rot_freq=rot_freq,
        fare_choice_vars=fare_choice_vars,
        interline_pax=interline_pax,
    )


def _market_by_id(markets: list[Market], market_id: str) -> Market:
    for m in markets:
        if m.market_id == market_id:
            return m
    raise KeyError(market_id)


def _extract_plan(
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
    connections: list[ConnectionCandidate],
    connect_pax: dict[str, cp_model.IntVar],
    rotations: list[RotationCandidate],
    rot_freq: dict[tuple[str, str], cp_model.IntVar],
    fare_choice_vars: dict[str, tuple[list[float], dict[int, cp_model.IntVar]]],
    interline_pax: dict[str, cp_model.IntVar],
) -> NetworkPlan:
    """Turn solver variable values back into domain objects."""
    from anos.forecast.demand import effective_demand

    ports = load_airports()
    market_plans: list[MarketPlan] = []
    fleet_used: dict[str, float] = {}

    connect_pax_by_market: dict[str, float] = {}
    for c in connections:
        if c.connection_id not in connect_pax:
            continue
        value = float(solver.Value(connect_pax[c.connection_id]))
        if value > 0:
            connect_pax_by_market[c.market_id] = connect_pax_by_market.get(c.market_id, 0.0) + value

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
        connecting = connect_pax_by_market.get(mid, 0.0)
        interlined = float(solver.Value(interline_pax[mid])) if mid in interline_pax else 0.0

        seats = sum(
            economics[(mid, t)].seats_round_trip * n for t, n in frequencies.items()
        )
        trip_cost = sum(
            economics[(mid, t)].cost_per_round_trip_usd * n
            for t, n in frequencies.items()
        )

        for t, n in frequencies.items():
            fleet_used[t] = (
                fleet_used.get(t, 0.0) + economics[(mid, t)].aircraft_hours_consumed * n
            )

        is_domestic = ports[market.origin].is_domestic and ports[market.destination].is_domestic

        fare_usd: float | None = None
        if mid in fare_choice_vars:
            fare_opts, fare_choice = fare_choice_vars[mid]
            for j, var in fare_choice.items():
                if solver.Value(var):
                    fare_usd = fare_opts[j]
                    break

        realised_fare = fare_usd if fare_usd is not None else market.avg_fare_usd
        revenue = (
            2 * carried * realised_fare
            + 2 * connecting * market.avg_fare_usd
            + 2 * interlined * params.interline_prorate_usd_per_pax
        )
        cost = (
            trip_cost
            + 2 * carried * params.cost_per_pax(domestic=is_domestic)
            + 2
            * connecting
            * (params.cost_per_pax(domestic=is_domestic) + params.connection_handling_cost_usd_per_pax)
        )

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
                connecting_pax_carried=connecting,
                fare_usd=fare_usd,
            )
        )

    rotations_flown: dict[str, int] = {}
    for r in rotations:
        total_loops = sum(
            solver.Value(var) for (rid, _t), var in rot_freq.items() if rid == r.rotation_id
        )
        if total_loops > 0:
            rotations_flown[r.rotation_id] = total_loops

    connections_used = sorted(mid for mid, v in connect_pax_by_market.items() if v > 0)
    interline_used = sorted(mid for mid, var in interline_pax.items() if solver.Value(var) > 0)

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
        connections_used=connections_used,
        rotations_flown=rotations_flown,
        interline_used=interline_used,
    )
