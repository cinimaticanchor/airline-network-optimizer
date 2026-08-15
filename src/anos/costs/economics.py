"""Unit economics of one round trip, for every (market, aircraft type) pairing.

Everything expensive or non-linear is computed here, once, before the solve. The
optimiser then only has to pick integer frequencies against precomputed per-trip
costs -- which is what keeps the model a linear MIP rather than something that needs
a global solver.

Cost of a round trip = fuel + crew + allocated ownership + airport/nav charges.
Passenger-driven cost (catering, distribution, commissions) scales with passengers
carried rather than trips flown, so it is applied in the objective instead.
"""

from __future__ import annotations

from anos.config import Params, load_params
from anos.data.loaders import load_aircraft_types, load_airports
from anos.models import (
    AircraftType,
    Airport,
    LegEconomics,
    Market,
    RotationCandidate,
    RotationEconomics,
    great_circle_km,
)


def _eligibility(
    market: Market, ac: AircraftType, origin: Airport, dest: Airport, distance_km: float, par: Params
) -> tuple[bool, str]:
    """Whether this type may fly this sector at all.

    Range is checked against a safety factor rather than the published figure: a
    published maximum range assumes a payload and winds you do not get on a revenue
    sector.
    """
    usable_range = ac.range_km * par.range_safety_factor
    if distance_km > usable_range:
        return False, f"sector {distance_km:,.0f} km exceeds usable range {usable_range:,.0f} km"

    if ac.is_widebody and not (origin.widebody_capable and dest.widebody_capable):
        limiting = origin.iata if not origin.widebody_capable else dest.iata
        return False, f"{limiting} cannot handle widebody aircraft"

    return True, ""


def leg_economics(
    market: Market,
    ac: AircraftType,
    *,
    airports: dict[str, Airport] | None = None,
    params: Params | None = None,
    fast_turn: bool = False,
    carbon_priced: bool = False,
) -> LegEconomics:
    """Cost, capacity and time consumed by one daily round trip.

    `fast_turn` swaps in `params.fast_turn_buffer_factor` for `turn_buffer_factor` --
    a minimum-connect-time programme that tightens planned ground time without
    changing the aircraft's technical minimum turn (`ac.min_turn_min`), freeing
    aircraft-hours for more departures at the same service level.

    `carbon_priced` folds a blended EU/UK ETS-style carbon cost into the round trip
    when either endpoint sits in `params.carbon_priced_countries`.
    """
    par = params or load_params()
    ports = airports or load_airports()
    origin, dest = ports[market.origin], ports[market.destination]

    distance_km = great_circle_km(origin.lat, origin.lon, dest.lat, dest.lon)
    eligible, reason = _eligibility(market, ac, origin, dest, distance_km, par)

    block_h_one_way = ac.block_hours(distance_km)
    block_h_round_trip = 2 * block_h_one_way

    # An aircraft is unavailable for other work while it sits on the ground, so
    # planned turn time is part of what a round trip costs in fleet capacity.
    buffer_factor = par.fast_turn_buffer_factor if fast_turn else par.turn_buffer_factor
    planned_turn_h = (ac.min_turn_min * buffer_factor) / 60.0
    aircraft_hours = block_h_round_trip + 2 * planned_turn_h

    # Fuel: cruise burn over block time, uplifted for reserves, plus ground burn
    # for each of the two cycles.
    fuel_kg = (
        ac.fuel_burn_kgh * block_h_round_trip * par.fuel_contingency_factor
        + 2 * par.taxi_burn_kg_per_cycle
    )
    fuel_cost = fuel_kg * par.fuel_price_usd_per_kg

    crew_cost = ac.crew_cost_per_block_hour_usd * block_h_round_trip

    # Ownership is a daily fixed cost; a round trip carries the share of the day it
    # occupies. A trip that uses half the aircraft's planned day carries half the
    # lease and capital charge.
    ownership_cost = (
        ac.ownership_cost_per_day_usd / ac.max_daily_util_h
    ) * aircraft_hours

    is_domestic = origin.is_domestic and dest.is_domestic
    charges = 2 * par.charges_per_departure(domestic=is_domestic, widebody=ac.is_widebody)
    for port in (origin, dest):
        if port.slot_constrained:
            charges += par.slot_surcharge_usd

    carbon_cost = 0.0
    if carbon_priced and {origin.country, dest.country} & par.carbon_priced_countries:
        co2_tonnes = (fuel_kg * par.co2_per_kg_fuel) / 1000.0
        carbon_cost = co2_tonnes * par.carbon_price_usd_per_tonne

    return LegEconomics(
        market_id=market.market_id,
        type_code=ac.code,
        distance_km=distance_km,
        block_hours_round_trip=block_h_round_trip,
        aircraft_hours_consumed=aircraft_hours,
        seats_round_trip=ac.seats,
        cost_per_round_trip_usd=fuel_cost + crew_cost + ownership_cost + charges + carbon_cost,
        eligible=eligible,
        ineligible_reason=reason,
    )


def build_economics(
    markets: list[Market] | None = None,
    *,
    params: Params | None = None,
    fast_turn: bool = False,
    carbon_priced: bool = False,
) -> dict[tuple[str, str], LegEconomics]:
    """Precompute economics for every (market, type) pairing.

    Returns a lookup keyed by (market_id, type_code) covering eligible pairings only;
    ineligible combinations are dropped so the optimiser never has to reason about
    them.
    """
    from anos.data.loaders import load_markets

    par = params or load_params()
    mkts = markets if markets is not None else load_markets()
    types = load_aircraft_types()
    ports = load_airports()

    out: dict[tuple[str, str], LegEconomics] = {}
    for market in mkts:
        for ac in types.values():
            econ = leg_economics(
                market, ac, airports=ports, params=par, fast_turn=fast_turn,
                carbon_priced=carbon_priced,
            )
            if econ.eligible:
                out[(market.market_id, ac.code)] = econ
    return out


def market_distance_km(market: Market) -> float:
    """Great-circle sector length for a market."""
    ports = load_airports()
    o, d = ports[market.origin], ports[market.destination]
    return great_circle_km(o.lat, o.lon, d.lat, d.lon)


def rotation_economics(
    candidate: RotationCandidate,
    ac: AircraftType,
    *,
    airports: dict[str, Airport] | None = None,
    params: Params | None = None,
) -> RotationEconomics:
    """Cost and aircraft-hours for flying one hub -> stop1 -> stop2 -> hub loop once.

    A loop shares one hub departure and three turns across both leg markets, instead
    of two separate round trips' four sectors and four turns -- it is a cheaper way
    to source the same two markets' frequency, not a new market of its own.
    """
    par = params or load_params()
    ports = airports or load_airports()
    hub = ports[candidate.hub]
    stop1 = ports[candidate.stops[0]]
    stop2 = ports[candidate.stops[1]]

    legs = [(hub, stop1), (stop1, stop2), (stop2, hub)]
    sector_km = [great_circle_km(a.lat, a.lon, b.lat, b.lon) for a, b in legs]

    eligible, reason = True, ""
    usable_range = ac.range_km * par.range_safety_factor
    longest = max(sector_km)
    if longest > usable_range:
        eligible = False
        reason = f"sector {longest:,.0f} km exceeds usable range {usable_range:,.0f} km"
    elif ac.is_widebody and not all(p.widebody_capable for p in (hub, stop1, stop2)):
        incapable = next(p.iata for p in (hub, stop1, stop2) if not p.widebody_capable)
        eligible = False
        reason = f"{incapable} cannot handle widebody aircraft"

    block_h = sum(ac.block_hours(d) for d in sector_km)
    planned_turn_h = (ac.min_turn_min * par.turn_buffer_factor) / 60.0
    aircraft_hours = block_h + 3 * planned_turn_h

    fuel_kg = (
        ac.fuel_burn_kgh * block_h * par.fuel_contingency_factor + 3 * par.taxi_burn_kg_per_cycle
    )
    fuel_cost = fuel_kg * par.fuel_price_usd_per_kg
    crew_cost = ac.crew_cost_per_block_hour_usd * block_h
    ownership_cost = (ac.ownership_cost_per_day_usd / ac.max_daily_util_h) * aircraft_hours

    charges = 0.0
    for a, b in legs:
        is_domestic = a.is_domestic and b.is_domestic
        charges += par.charges_per_departure(domestic=is_domestic, widebody=ac.is_widebody)
        if a.slot_constrained:
            charges += par.slot_surcharge_usd

    return RotationEconomics(
        rotation_id=candidate.rotation_id,
        type_code=ac.code,
        aircraft_hours_consumed=aircraft_hours,
        cost_usd=fuel_cost + crew_cost + ownership_cost + charges,
        eligible=eligible,
        ineligible_reason=reason,
    )


def build_rotation_economics(
    candidates: list[RotationCandidate],
    *,
    params: Params | None = None,
) -> dict[tuple[str, str], RotationEconomics]:
    """Precompute economics for every (rotation, type) pairing, eligible ones only."""
    par = params or load_params()
    types = load_aircraft_types()
    ports = load_airports()

    out: dict[tuple[str, str], RotationEconomics] = {}
    for candidate in candidates:
        for ac in types.values():
            econ = rotation_economics(candidate, ac, airports=ports, params=par)
            if econ.eligible:
                out[(candidate.rotation_id, ac.code)] = econ
    return out
