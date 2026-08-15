"""Connection candidates (banking) and rotation candidates (circular routing).

Both are instances of one idea: let two legs that are already flown (or two markets
that are already thin) share capacity to serve something the point-to-point model
cannot see on its own. See the module docstrings on `ConnectionCandidate` and
`RotationCandidate` in `anos.models` for what each represents; this module only
builds the candidate lists `anos.optimize.fleet_assignment` consumes.
"""

from __future__ import annotations

from datetime import date

from anos.models import ConnectionCandidate, Market, NetworkPlan, RotationCandidate


def build_connection_candidates(
    markets: list[Market], hubs: list[str]
) -> list[ConnectionCandidate]:
    """Every market that could plausibly be served one-stop via one of `hubs`.

    A candidate exists whenever markets.csv already has both the feeder leg
    (hub-origin) and the trunk leg (hub-destination) as their own rows -- hub-first,
    which is the only form present in the data. This deliberately does not filter to
    markets that are currently unserved: a connection is just an extra capacity
    option the solver may or may not find worth using alongside the nonstop.
    """
    by_id = {m.market_id: m for m in markets}
    candidates: list[ConnectionCandidate] = []
    for hub in hubs:
        for market in markets:
            if hub in (market.origin, market.destination):
                continue  # would be the direct hub market itself, not a connection
            feeder_id = f"{hub}-{market.origin}"
            trunk_id = f"{hub}-{market.destination}"
            if feeder_id in by_id and trunk_id in by_id:
                candidates.append(
                    ConnectionCandidate(
                        connection_id=f"{market.origin}-{hub}-{market.destination}",
                        market_id=market.market_id,
                        hub=hub,
                        feeder_market_id=feeder_id,
                        trunk_market_id=trunk_id,
                    )
                )
    return candidates


def build_rotation_candidates(
    markets: list[Market], squad: list[str], hubs: list[str], *, max_stops: int = 2
) -> list[RotationCandidate]:
    """Rotation loops (hub -> stop1 -> stop2 -> hub) built only from `squad` markets.

    `squad` is an explicit, small list of market_ids (typically today's low-load-
    factor legs -- see `low_load_factor_squad`), not searched for. Only 2-stop loops
    are supported (`max_stops` is currently required to be 2); candidates are every
    *ordered* pair of squad markets that share a hub, giving exactly k*(k-1) loops for
    a hub with k squad spokes -- never a general search over the network.
    """
    if max_stops != 2:
        raise NotImplementedError("only 2-stop rotations are supported")

    by_id = {m.market_id: m for m in markets}
    squad_markets = [by_id[mid] for mid in squad if mid in by_id]

    candidates: list[RotationCandidate] = []
    for hub in hubs:
        spokes = [m for m in squad_markets if m.origin == hub]
        for a in spokes:
            for b in spokes:
                if a.market_id == b.market_id:
                    continue
                candidates.append(
                    RotationCandidate(
                        rotation_id=f"{hub}-{a.destination}-{b.destination}",
                        hub=hub,
                        stops=(a.destination, b.destination),
                        leg_market_ids=(a.market_id, b.market_id),
                    )
                )
    return candidates


def low_load_factor_squad(plan: NetworkPlan, *, threshold: float = 0.50) -> list[str]:
    """Market ids currently flown at or below `threshold` load factor.

    This is "the squad": aircraft-hours today's plan already spends, spent on markets
    too thin to justify a dedicated out-and-back on their own.
    """
    return sorted(
        p.market_id
        for p in plan.market_plans
        if p.total_frequency > 0 and p.load_factor <= threshold
    )


def solve_extended_network(
    target: date,
    *,
    month: int | None = None,
    hubs: list[str],
    fast_turn: bool = True,
    enable_fare_elasticity: bool = True,
    enable_connections: bool = True,
    enable_rotations: bool = True,
    rotation_lf_threshold: float = 0.50,
    baseline: NetworkPlan | None = None,
) -> tuple[NetworkPlan, NetworkPlan]:
    """Two-pass workflow shared by the CLI and tests: solve the baseline, find its
    low-load-factor squad, then re-solve with every requested extension switched on.

    Returns `(baseline, extended)`. Passing `baseline` in skips the first solve (the
    caller may already have one, e.g. from `anos solve`).
    """
    from anos.optimize.fleet_assignment import solve_network

    base = baseline or solve_network(target, month=month)
    squad = low_load_factor_squad(base, threshold=rotation_lf_threshold)

    extended = solve_network(
        target,
        month=month,
        fast_turn=fast_turn,
        enable_fare_elasticity=enable_fare_elasticity,
        enable_connections=enable_connections,
        hubs=hubs,
        enable_rotations=enable_rotations,
        rotation_squad=squad if enable_rotations else None,
    )
    return base, extended
