"""Bucket arithmetic for the time-space network fleet-assignment core.

Sibling to `anos.network.dayparts` (that module's coarse 4-bucket clock stays owned
by hub-connection feasibility checking and is untouched by this one). This clock
exists to make real fleet-count flow-balance possible: a departure consumes one unit
of an aircraft type's count at the origin station and returns it at the destination
station only once block time *and* planned turn time have passed -- "available
again", not merely landed.

**Buckets are on one shared reference clock, not each station's local time.**
`anos.network.dayparts` converts through each airport's `timezone_offset_h` because
its buckets represent a single *location*'s (the hub's) local wall clock, which is
all a layover needs. This module's fleet-count cap sums aircraft *across different
stations at the same instant* -- for that sum to mean anything, every station's
bucket index must refer to the same instant. Mixing in per-station local-time
conversion here would silently break that: station A's bucket 3 and station B's
bucket 3 would be different moments, and "aircraft in the network right now" would
add together aircraft from two different times. So `avail_bucket` below is pure
elapsed-time arithmetic on one axis; local wall-clock realism (which the model
doesn't need for capacity feasibility -- see `anos.optimize.timespace_assignment`'s
stated non-goals) is deliberately out of scope.

`config` defaults to `data/timespace.yaml`, kept deliberately coarser than a real
airline's minute-level schedule (see `anos.optimize.timespace_assignment`'s module
docstring for the variable-count trade-off that sizing this trades against).
"""

from __future__ import annotations

from anos.data.loaders import load_timespace_config


def n_buckets(config: dict | None = None) -> int:
    return int((config or load_timespace_config())["buckets"])


def _bucket_hours(config: dict | None = None) -> float:
    return 24.0 / n_buckets(config)


def bucket_of(hour: float, config: dict | None = None) -> int:
    """Which bucket a reference-clock hour (may be outside [0,24), e.g. after
    arithmetic that crosses a cycle boundary) falls in."""
    bucket_hours = _bucket_hours(config)
    return int((hour % 24.0) // bucket_hours) % n_buckets(config)


def bucket_midpoint_hour(bucket: int, config: dict | None = None) -> float:
    bucket_hours = _bucket_hours(config)
    return bucket * bucket_hours + bucket_hours / 2.0


def avail_bucket(
    dep_bucket: int, duration_h: float, config: dict | None = None
) -> tuple[int, float]:
    """Which bucket the aircraft is available again in, given it departs `dep_bucket`.

    `duration_h` is one-way block time *plus* planned turn time -- when the aircraft
    can next be used for something else, not when it lands. Returns
    `(available_bucket, duration_in_buckets)`; the caller is responsible for treating
    `duration_in_buckets >= n_buckets` as infeasible (a leg that can't complete within
    one cyclic day at this bucket width).
    """
    dep_hour = bucket_midpoint_hour(dep_bucket, config)
    avail_hour = dep_hour + duration_h
    duration_in_buckets = duration_h / _bucket_hours(config)
    return bucket_of(avail_hour, config), duration_in_buckets


def straddles_anchor(dep_bucket: int, avail_bucket_idx: int, anchor: int, buckets: int) -> bool:
    """Whether the half-open cyclic interval [dep_bucket, avail_bucket_idx) contains
    `anchor` -- i.e. whether an aircraft departing `dep_bucket` and not available
    again until `avail_bucket_idx` is "in the network" (not on the ground) at the
    anchor cross-section.

    `anchor == dep_bucket` counts as inside (the aircraft has just left the ground);
    `anchor == avail_bucket_idx` does not (it is available again, i.e. effectively
    back on the ground, by then). A zero-length interval (`dep_bucket ==
    avail_bucket_idx`) straddles nothing.

    See `anos.optimize.timespace_assignment`'s module docstring for why counting
    tokens at *any single* fixed cross-section correctly enforces the fleet-count
    cap in a flow-conserving network -- `anchor`'s specific value never changes the
    answer, only how readable debugging output is.
    """
    span = (avail_bucket_idx - dep_bucket) % buckets
    if span == 0:
        return False
    offset = (anchor - dep_bucket) % buckets
    return offset < span
