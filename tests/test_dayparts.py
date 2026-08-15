"""The coarse daypart-bucket clock connection feasibility is checked against."""

from __future__ import annotations

from anos.models import Airport
from anos.network.dayparts import (
    arrival_daypart,
    connection_feasible,
    connection_layover_hours,
    daypart_buckets,
    daypart_of,
    feeder_arrival_daypart,
)


def make_airport(**overrides) -> Airport:
    base = dict(
        iata="TST",
        name="Test",
        city="Test",
        country="IN",
        lat=0.0,
        lon=0.0,
        widebody_capable=True,
        slot_constrained=False,
        daily_slot_cap=100,
        curfew=False,
        timezone_offset_h=0.0,
    )
    base.update(overrides)
    return Airport(**base)


TOPOLOGY = {
    "daypart_buckets": 4,
    "connection_window": {"min_layover_h": 1.0, "max_layover_h": 8.0},
}


def test_default_topology_uses_four_buckets():
    assert daypart_buckets() == 4


def test_daypart_of_bucket_boundaries():
    assert daypart_of(0.0, topology=TOPOLOGY) == 0
    assert daypart_of(5.99, topology=TOPOLOGY) == 0
    assert daypart_of(6.0, topology=TOPOLOGY) == 1
    assert daypart_of(15.0, topology=TOPOLOGY) == 2
    assert daypart_of(21.0, topology=TOPOLOGY) == 3


def test_daypart_of_wraps_past_midnight():
    assert daypart_of(25.0, topology=TOPOLOGY) == daypart_of(1.0, topology=TOPOLOGY)


def test_arrival_daypart_short_hop_same_timezone_stays_in_bucket():
    origin = make_airport(iata="A", timezone_offset_h=0.0)
    dest = make_airport(iata="B", timezone_offset_h=0.0)
    # Departs bucket 0 (midpoint 03:00), 2h block -> lands 05:00, still bucket 0.
    assert arrival_daypart(0, 2.0, origin, dest, topology=TOPOLOGY) == 0


def test_arrival_daypart_long_hop_crosses_into_next_bucket():
    origin = make_airport(iata="A", timezone_offset_h=0.0)
    dest = make_airport(iata="B", timezone_offset_h=0.0)
    # Departs bucket 0 (03:00), 5h block -> lands 08:00, bucket 1.
    assert arrival_daypart(0, 5.0, origin, dest, topology=TOPOLOGY) == 1


def test_arrival_daypart_accounts_for_timezone_offset():
    origin = make_airport(iata="A", timezone_offset_h=0.0)
    dest_east = make_airport(iata="B", timezone_offset_h=6.0)
    dest_same = make_airport(iata="C", timezone_offset_h=0.0)
    # Same block time, but the eastward destination's clock is 6h ahead.
    assert arrival_daypart(0, 1.0, origin, dest_east, topology=TOPOLOGY) != arrival_daypart(
        0, 1.0, origin, dest_same, topology=TOPOLOGY
    )


def test_feeder_arrival_daypart_uses_reference_type_block_hours():
    origin = make_airport(iata="A", timezone_offset_h=0.0)
    hub = make_airport(iata="B", timezone_offset_h=0.0)
    # A short domestic sector should land in the same or the next bucket, never wrap
    # the whole clock.
    d = feeder_arrival_daypart(0, 600.0, origin, hub, topology=TOPOLOGY)
    assert d in (0, 1)


def test_connection_layover_hours_same_bucket_is_zero():
    assert connection_layover_hours(1, 1, topology=TOPOLOGY) == 0.0


def test_connection_layover_hours_wraps_past_midnight():
    # Feeder lands in the last bucket (mid 21:00), trunk departs the first (mid 03:00
    # the next day) -- six hours later, not eighteen hours earlier.
    layover = connection_layover_hours(3, 0, topology=TOPOLOGY)
    assert layover == 6.0


def test_connection_feasible_within_window():
    assert connection_feasible(1, 2, topology=TOPOLOGY) is True  # 6h layover


def test_connection_infeasible_too_short():
    assert connection_feasible(1, 1, topology=TOPOLOGY) is False  # 0h layover


def test_connection_infeasible_too_long():
    assert connection_feasible(0, 3, topology=TOPOLOGY) is False  # 18h layover
