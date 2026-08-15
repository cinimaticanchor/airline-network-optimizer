"""A coarse clock for a model that otherwise has none.

Nothing else in this codebase represents departure/arrival time -- `solve_network`
picks an aggregate daily frequency per market, not individual flight times. Making
"banking" (a connection being schedule-feasible) checkable needs *some* notion of
when in the day a market's flights operate, without building a real minute-level
schedule engine.

The compromise: split the day into a handful of equal buckets (`daypart_buckets` in
hub_topology.yaml). A market's daypart is a solver decision (see
`anos.optimize.fleet_assignment`), not a curated timetable entry. A feeder leg's
arrival daypart is computed formulaically from its departure daypart, one-way block
time, and the two airports' timezone offsets -- both already present in the reference
data. This is deliberately too coarse to reason about individual flight times, and
precise enough to make "does this connection make sense at all" a checkable fact
instead of an assumption.
"""

from __future__ import annotations

from anos.data.loaders import load_aircraft_types, load_hub_topology
from anos.models import AircraftType, Airport

# Used only to estimate a feeder leg's block time for daypart bucketing -- not for
# costing or eligibility, which always use the actual assigned aircraft type's
# economics. Nearly every domestic feeder leg in this network is narrowbody, and at
# bucket-width granularity the choice of narrowbody type barely moves the answer.
REFERENCE_FEEDER_TYPE = "A320neo"


def daypart_buckets(topology: dict | None = None) -> int:
    return int((topology or load_hub_topology())["daypart_buckets"])


def _bucket_hours(topology: dict | None = None) -> float:
    return 24.0 / daypart_buckets(topology)


def daypart_of(local_hour: float, *, topology: dict | None = None) -> int:
    """Which bucket a local clock hour (may be outside [0,24), e.g. after wraparound) falls in."""
    bucket_hours = _bucket_hours(topology)
    return int((local_hour % 24.0) // bucket_hours) % daypart_buckets(topology)


def _bucket_midpoint_hour(bucket: int, *, topology: dict | None = None) -> float:
    bucket_hours = _bucket_hours(topology)
    return bucket * bucket_hours + bucket_hours / 2.0


def arrival_daypart(
    departure_daypart: int,
    block_hours_one_way: float,
    origin: Airport,
    dest: Airport,
    *,
    topology: dict | None = None,
) -> int:
    """Which bucket a leg lands in, given the bucket it (notionally) departs in.

    Assumes departure happens at the departure bucket's midpoint local time -- the
    only assumption a bucket model can make about *when inside the bucket* a flight
    sits.
    """
    dep_local = _bucket_midpoint_hour(departure_daypart, topology=topology)
    dep_utc = dep_local - origin.timezone_offset_h
    arr_utc = dep_utc + block_hours_one_way
    arr_local = arr_utc + dest.timezone_offset_h
    return daypart_of(arr_local, topology=topology)


def feeder_arrival_daypart(
    departure_daypart: int,
    distance_km: float,
    origin: Airport,
    hub: Airport,
    *,
    types: dict[str, AircraftType] | None = None,
    topology: dict | None = None,
) -> int:
    """Convenience wrapper: arrival daypart at `hub` for a feeder leg of this distance."""
    ac = (types or load_aircraft_types())[REFERENCE_FEEDER_TYPE]
    return arrival_daypart(
        departure_daypart, ac.block_hours(distance_km), origin, hub, topology=topology
    )


def connection_layover_hours(
    feeder_arrival_daypart: int, trunk_departure_daypart: int, *, topology: dict | None = None
) -> float:
    """Layover between a feeder's (bucket-midpoint) arrival and a trunk's departure."""
    arr_local = _bucket_midpoint_hour(feeder_arrival_daypart, topology=topology)
    dep_local = _bucket_midpoint_hour(trunk_departure_daypart, topology=topology)
    return (dep_local - arr_local) % 24.0


def connection_feasible(
    feeder_arrival_daypart: int,
    trunk_departure_daypart: int,
    *,
    topology: dict | None = None,
) -> bool:
    """Whether a feeder landing in one bucket can realistically bank onto a trunk
    departing in another, given the configured connection window."""
    window = (topology or load_hub_topology())["connection_window"]
    layover = connection_layover_hours(
        feeder_arrival_daypart, trunk_departure_daypart, topology=topology
    )
    return window["min_layover_h"] <= layover <= window["max_layover_h"]
